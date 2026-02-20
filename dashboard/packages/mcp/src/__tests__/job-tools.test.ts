import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { TidalApiClient, ApiResult } from "../http-client.js";
import { handleGetJobs, handleGetJobLogs } from "../tools/job-tools.js";

// ---------------------------------------------------------------------------
// Mock client factory
// ---------------------------------------------------------------------------

function okClient<T>(data: T): TidalApiClient {
  return {
    get: async () => ({ ok: true, data, status: 200 }) as ApiResult<never>,
    post: async () => ({ ok: true, data: {}, status: 200 }) as ApiResult<never>,
    put: async () => ({ ok: true, data: {}, status: 200 }) as ApiResult<never>,
  };
}

function errClient(status: number, error: string): TidalApiClient {
  return {
    get: async () => ({ ok: false, error, status }) as ApiResult<never>,
    post: async () => ({ ok: false, error, status }) as ApiResult<never>,
    put: async () => ({ ok: false, error, status }) as ApiResult<never>,
  };
}

// ---------------------------------------------------------------------------
// handleGetJobs
// ---------------------------------------------------------------------------

describe("handleGetJobs", () => {
  it("returns job list", async () => {
    const data = {
      jobs: [
        {
          jobId: "job-1",
          type: "lm-training",
          status: "running",
          provider: "local",
          config: { type: "lm-training", plugin: "tidal", configPath: "x.yaml" },
          createdAt: 1700000000000,
          updatedAt: 1700000000000,
        },
      ],
    };
    const result = await handleGetJobs(okClient(data));

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.jobs.length, 1);
    assert.equal(parsed.jobs[0].jobId, "job-1");
    assert.equal(parsed.jobs[0].status, "running");
  });

  it("returns error on failure", async () => {
    const result = await handleGetJobs(errClient(500, "Database error"));
    assert.equal(result.isError, true);
    assert.match(result.content[0].text as string, /Database error/);
  });
});

// ---------------------------------------------------------------------------
// handleGetJobLogs
// ---------------------------------------------------------------------------

describe("handleGetJobLogs", () => {
  it("returns job logs", async () => {
    const data = {
      jobId: "job-1",
      lines: [
        { timestamp: 1000, stream: "stdout", line: "Epoch 1/10" },
        { timestamp: 1001, stream: "stderr", line: "Warning: low GPU memory" },
      ],
      totalLines: 2,
    };
    const result = await handleGetJobLogs(okClient(data), { jobId: "job-1" });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.lines.length, 2);
    assert.equal(parsed.lines[0].line, "Epoch 1/10");
  });

  it("passes offset and limit query params", async () => {
    let capturedQuery: Record<string, unknown> | undefined;
    const client: TidalApiClient = {
      get: async (_path, query) => {
        capturedQuery = query as Record<string, unknown>;
        return {
          ok: true,
          data: { jobId: "job-1", lines: [], totalLines: 0 },
          status: 200,
        } as ApiResult<never>;
      },
      post: async () => ({ ok: true, data: {}, status: 200 }) as ApiResult<never>,
      put: async () => ({ ok: true, data: {}, status: 200 }) as ApiResult<never>,
    };

    await handleGetJobLogs(client, { jobId: "job-1", offset: 50, limit: 100 });

    assert.equal(capturedQuery?.offset, 50);
    assert.equal(capturedQuery?.limit, 100);
  });

  it("returns error on missing job", async () => {
    const result = await handleGetJobLogs(errClient(404, "Job not found"), {
      jobId: "missing",
    });
    assert.equal(result.isError, true);
  });
});
