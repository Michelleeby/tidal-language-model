import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { TidalApiClient, ApiResult } from "../http-client.js";
import {
  handleListPatterns,
  handleGenerateReport,
} from "../tools/report-tools.js";

// ---------------------------------------------------------------------------
// Mock client factory
// ---------------------------------------------------------------------------

function mockClient(
  getResult: ApiResult<unknown>,
  postResult?: ApiResult<unknown>,
): TidalApiClient {
  return {
    get: async () => getResult as ApiResult<never>,
    post: async () => (postResult ?? getResult) as ApiResult<never>,
  };
}

function okPostClient<T>(data: T): TidalApiClient {
  return mockClient(
    { ok: true, data: {}, status: 200 },
    { ok: true, data, status: 201 },
  );
}

function errPostClient(status: number, error: string): TidalApiClient {
  return mockClient(
    { ok: true, data: {}, status: 200 },
    { ok: false, error, status },
  );
}

// ---------------------------------------------------------------------------
// handleListPatterns
// ---------------------------------------------------------------------------

describe("handleListPatterns", () => {
  it("returns all 4 patterns", async () => {
    const result = await handleListPatterns();

    assert.equal(result.isError, undefined);
    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.patterns.length, 4);

    const names = parsed.patterns.map((p: { name: string }) => p.name);
    assert.ok(names.includes("experiment-overview"));
    assert.ok(names.includes("rl-analysis"));
    assert.ok(names.includes("trajectory-report"));
    assert.ok(names.includes("full-report"));
  });

  it("each pattern has a name and description", async () => {
    const result = await handleListPatterns();
    const parsed = JSON.parse(result.content[0].text as string);

    for (const pattern of parsed.patterns) {
      assert.ok(pattern.name, "pattern should have a name");
      assert.ok(pattern.description, "pattern should have a description");
    }
  });
});

// ---------------------------------------------------------------------------
// handleGenerateReport
// ---------------------------------------------------------------------------

describe("handleGenerateReport", () => {
  it("returns report on success", async () => {
    const fakeReport = {
      report: {
        id: "rpt-1",
        userId: null,
        title: "experiment-overview — exp-1",
        blocks: [{ type: "heading" }],
        createdAt: 1000,
        updatedAt: 1000,
      },
    };
    const client = okPostClient(fakeReport);

    const result = await handleGenerateReport(client, {
      pattern: "experiment-overview",
      experimentId: "exp-1",
    });

    assert.equal(result.isError, undefined);
    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.report.id, "rpt-1");
  });

  it("returns error on API failure", async () => {
    const client = errPostClient(500, "Internal server error");

    const result = await handleGenerateReport(client, {
      pattern: "experiment-overview",
      experimentId: "exp-1",
    });

    assert.equal(result.isError, true);
    assert.match(result.content[0].text as string, /Internal server error/);
  });

  it("passes githubLogin and title through to API", async () => {
    let capturedBody: unknown;
    const client: TidalApiClient = {
      get: async () => ({ ok: true, data: {}, status: 200 }) as ApiResult<never>,
      post: async (_path, body) => {
        capturedBody = body;
        return {
          ok: true,
          data: { report: { id: "rpt-2", blocks: [] } },
          status: 201,
        } as ApiResult<never>;
      },
    };

    await handleGenerateReport(client, {
      pattern: "full-report",
      experimentId: "exp-99",
      title: "My Report",
      githubLogin: "octocat",
    });

    assert.deepEqual(capturedBody, {
      pattern: "full-report",
      experimentId: "exp-99",
      title: "My Report",
      githubLogin: "octocat",
    });
  });
});
