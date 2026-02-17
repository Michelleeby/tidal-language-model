import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { TidalApiClient, ApiResult } from "../http-client.js";
import {
  handleListExperiments,
  handleGetMetrics,
  handleGetRlMetrics,
  handleGetStatus,
  handleGetCheckpoints,
  handleGetEvaluation,
  handleGetAblation,
  handleGetGpuInstance,
} from "../tools/experiment-tools.js";

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

function okClient<T>(data: T): TidalApiClient {
  return mockClient({ ok: true, data, status: 200 });
}

function errClient(status: number, error: string): TidalApiClient {
  return mockClient({ ok: false, error, status });
}

// ---------------------------------------------------------------------------
// handleListExperiments
// ---------------------------------------------------------------------------

describe("handleListExperiments", () => {
  it("returns experiments as JSON text", async () => {
    const data = {
      experiments: [
        {
          id: "exp-1",
          path: "/experiments/exp-1",
          created: 1700000000000,
          hasRLMetrics: false,
          hasEvaluation: true,
          hasAblation: false,
          status: null,
          checkpoints: ["epoch_1.pth"],
        },
      ],
    };
    const result = await handleListExperiments(okClient(data));

    assert.equal(result.content.length, 1);
    assert.equal(result.content[0].type, "text");
    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.experiments.length, 1);
    assert.equal(parsed.experiments[0].id, "exp-1");
  });

  it("returns error on failure", async () => {
    const result = await handleListExperiments(
      errClient(500, "Server down"),
    );

    assert.equal(result.isError, true);
    assert.match(result.content[0].text as string, /Server down/);
  });
});

// ---------------------------------------------------------------------------
// handleGetMetrics
// ---------------------------------------------------------------------------

describe("handleGetMetrics", () => {
  it("returns metrics with query params", async () => {
    const data = {
      expId: "exp-1",
      points: [{ step: 1, timestamp: 1000, "Losses/Total": 2.5 }],
      totalPoints: 1,
      originalCount: 1,
      downsampled: false,
    };
    // Verify the client receives query params by capturing them
    let capturedPath = "";
    let capturedQuery: Record<string, unknown> | undefined;
    const client: TidalApiClient = {
      get: async (path, query) => {
        capturedPath = path;
        capturedQuery = query as Record<string, unknown>;
        return { ok: true, data, status: 200 } as ApiResult<never>;
      },
      post: async () => ({ ok: true, data: {}, status: 200 }) as ApiResult<never>,
    };

    const result = await handleGetMetrics(client, {
      expId: "exp-1",
      mode: "recent",
      window: 100,
      maxPoints: 500,
    });

    assert.equal(capturedPath, "/api/experiments/exp-1/metrics");
    assert.equal(capturedQuery?.mode, "recent");
    assert.equal(capturedQuery?.window, 100);
    assert.equal(capturedQuery?.maxPoints, 500);
    assert.equal(result.isError, undefined);
    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.points.length, 1);
  });

  it("handles error response", async () => {
    const result = await handleGetMetrics(
      errClient(404, "Experiment not found"),
      { expId: "missing" },
    );

    assert.equal(result.isError, true);
    assert.match(result.content[0].text as string, /Experiment not found/);
  });
});

// ---------------------------------------------------------------------------
// handleGetRlMetrics
// ---------------------------------------------------------------------------

describe("handleGetRlMetrics", () => {
  it("returns RL metrics", async () => {
    const data = {
      expId: "exp-1",
      metrics: {
        global_step: 100,
        timestamp: 1000,
        history: {
          episode_rewards: [0.5, 0.6],
          policy_loss: [0.1],
          value_loss: [0.2],
          entropy: [0.3],
        },
      },
    };
    const result = await handleGetRlMetrics(okClient(data), {
      expId: "exp-1",
    });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.metrics.global_step, 100);
  });

  it("returns null metrics gracefully", async () => {
    const data = { expId: "exp-1", metrics: null };
    const result = await handleGetRlMetrics(okClient(data), {
      expId: "exp-1",
    });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.metrics, null);
    assert.equal(result.isError, undefined);
  });
});

// ---------------------------------------------------------------------------
// handleGetStatus
// ---------------------------------------------------------------------------

describe("handleGetStatus", () => {
  it("returns training status", async () => {
    const data = {
      expId: "exp-1",
      status: {
        status: "training",
        last_update: 1000,
        current_step: 50,
        total_steps: 100,
      },
    };
    const result = await handleGetStatus(okClient(data), { expId: "exp-1" });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.status.status, "training");
    assert.equal(parsed.status.current_step, 50);
  });
});

// ---------------------------------------------------------------------------
// handleGetCheckpoints
// ---------------------------------------------------------------------------

describe("handleGetCheckpoints", () => {
  it("returns checkpoint list", async () => {
    const data = {
      expId: "exp-1",
      checkpoints: [
        {
          filename: "epoch_1.pth",
          path: "/experiments/exp-1/epoch_1.pth",
          sizeBytes: 1024000,
          modified: 1700000000000,
          phase: "foundational",
          epoch: 1,
        },
      ],
    };
    const result = await handleGetCheckpoints(okClient(data), {
      expId: "exp-1",
    });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.checkpoints.length, 1);
    assert.equal(parsed.checkpoints[0].filename, "epoch_1.pth");
  });
});

// ---------------------------------------------------------------------------
// handleGetEvaluation
// ---------------------------------------------------------------------------

describe("handleGetEvaluation", () => {
  it("returns evaluation results", async () => {
    const data = {
      expId: "exp-1",
      results: {
        perplexity: 42.5,
        samples: [{ prompt: "Once", generated: "Once upon a time" }],
      },
    };
    const result = await handleGetEvaluation(okClient(data), {
      expId: "exp-1",
    });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.results.perplexity, 42.5);
  });

  it("handles null evaluation gracefully", async () => {
    const data = { expId: "exp-1", results: null };
    const result = await handleGetEvaluation(okClient(data), {
      expId: "exp-1",
    });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.results, null);
    assert.equal(result.isError, undefined);
  });
});

// ---------------------------------------------------------------------------
// handleGetAblation
// ---------------------------------------------------------------------------

describe("handleGetAblation", () => {
  it("returns ablation results", async () => {
    const data = {
      expId: "exp-1",
      results: {
        learned: {
          mean_reward: 0.8,
          std_reward: 0.1,
          mean_diversity: 0.6,
          mean_perplexity: 30,
        },
      },
    };
    const result = await handleGetAblation(okClient(data), {
      expId: "exp-1",
    });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.results.learned.mean_reward, 0.8);
  });

  it("handles null ablation gracefully", async () => {
    const data = { expId: "exp-1", results: null };
    const result = await handleGetAblation(okClient(data), {
      expId: "exp-1",
    });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.results, null);
    assert.equal(result.isError, undefined);
  });
});

// ---------------------------------------------------------------------------
// handleGetGpuInstance
// ---------------------------------------------------------------------------

describe("handleGetGpuInstance", () => {
  it("returns GPU instance metadata", async () => {
    const data = {
      expId: "exp-1",
      instance: {
        instanceId: 31562809,
        offerId: 500,
        gpuName: "RTX A6000",
        costPerHour: 0.65,
        capturedAt: 1700000000000,
      },
    };
    const result = await handleGetGpuInstance(okClient(data), {
      expId: "exp-1",
    });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.instance.instanceId, 31562809);
    assert.equal(parsed.instance.gpuName, "RTX A6000");
  });

  it("handles null instance gracefully", async () => {
    const data = { expId: "exp-1", instance: null };
    const result = await handleGetGpuInstance(okClient(data), {
      expId: "exp-1",
    });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.instance, null);
    assert.equal(result.isError, undefined);
  });
});
