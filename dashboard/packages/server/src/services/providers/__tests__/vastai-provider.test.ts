import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import type { TrainingJob } from "@tidal/shared";
import { VastAIProvider, type VastAIProviderConfig } from "../vastai-provider.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeJob(overrides: Partial<TrainingJob> = {}): TrainingJob {
  return {
    jobId: "job-" + Math.random().toString(36).slice(2, 8),
    type: "rl-training",
    status: "provisioning",
    provider: "vastai",
    config: {
      type: "rl-training",
      plugin: "tidal",
      configPath: "plugins/tidal/configs/base_config.yaml",
    },
    createdAt: Date.now(),
    updatedAt: Date.now(),
    ...overrides,
  };
}

function silentLogger() {
  return {
    info() {},
    warn() {},
    error() {},
    debug() {},
    fatal() {},
    trace() {},
    child() {
      return silentLogger();
    },
  } as unknown as VastAIProviderConfig["log"];
}

function makeProvider(overrides: Partial<VastAIProviderConfig> = {}): VastAIProvider {
  return new VastAIProvider({
    apiKey: "test-key",
    dashboardUrl: "http://localhost:4400",
    authToken: "test-token",
    repoUrl: "https://github.com/test/repo.git",
    log: silentLogger(),
    ...overrides,
  });
}

interface FakeOffer {
  id: number;
  gpu_name: string;
  gpu_ram: number;
  dph_total: number;
  rentable: boolean;
  num_gpus: number;
}

function makeOffer(id: number, dph = 0.5): FakeOffer {
  return {
    id,
    gpu_name: "RTX 4090",
    gpu_ram: 24000,
    dph_total: dph,
    rentable: true,
    num_gpus: 1,
  };
}

/**
 * Install a fake `fetch` that routes VastAI API calls to handlers.
 * - `searchResponse`: controls what the /bundles search returns
 * - `createResponses`: a Map from offer id → Response for PUT /asks/:id/
 */
function installFakeFetch(opts: {
  offers: FakeOffer[];
  createResponses: Map<number, { ok: boolean; status: number; body: unknown }>;
}) {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = (async (
    input: string | URL | Request,
    init?: RequestInit,
  ): Promise<Response> => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;

    // Search offers endpoint
    if (url.includes("/bundles")) {
      return new Response(JSON.stringify({ offers: opts.offers }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Create instance endpoint — match /asks/:id/
    const askMatch = url.match(/\/asks\/(\d+)\//);
    if (askMatch && init?.method === "PUT") {
      const offerId = Number(askMatch[1]);
      const resp = opts.createResponses.get(offerId);
      if (!resp) {
        return new Response("unexpected offer id", { status: 500 });
      }
      return new Response(JSON.stringify(resp.body), {
        status: resp.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Fallback — should not be reached in tests
    return originalFetch(input, init as RequestInit);
  }) as typeof globalThis.fetch;

  return () => {
    globalThis.fetch = originalFetch;
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("VastAIProvider.provision()", () => {
  let restoreFetch: (() => void) | undefined;

  afterEach(() => {
    restoreFetch?.();
    restoreFetch = undefined;
  });

  it("succeeds on first try", async () => {
    const offers = [makeOffer(100)];
    restoreFetch = installFakeFetch({
      offers,
      createResponses: new Map([
        [100, { ok: true, status: 200, body: { new_contract: 9001 } }],
      ]),
    });

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, true);
    assert.equal(result.meta?.instanceId, 9001);
    assert.equal(result.meta?.offerId, 100);
  });

  it("retries next offer when first offer is unavailable", async () => {
    const offers = [makeOffer(200, 0.4), makeOffer(201, 0.5), makeOffer(202, 0.6)];

    restoreFetch = installFakeFetch({
      offers,
      createResponses: new Map([
        [
          200,
          {
            ok: false,
            status: 400,
            body: {
              success: false,
              error: "invalid_args",
              msg: "error 404/3603: no_such_ask Instance type by id 200 is not available.",
              ask_id: 200,
            },
          },
        ],
        [201, { ok: true, status: 200, body: { new_contract: 9002 } }],
      ]),
    });

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, true);
    assert.equal(result.meta?.instanceId, 9002);
    assert.equal(result.meta?.offerId, 201);
  });

  it("fails when all offers are unavailable", async () => {
    const offers = [makeOffer(300), makeOffer(301), makeOffer(302)];

    const staleResponse = (id: number) => ({
      ok: false as const,
      status: 400,
      body: {
        success: false,
        error: "invalid_args",
        msg: `error 404/3603: no_such_ask Instance type by id ${id} is not available.`,
        ask_id: id,
      },
    });

    restoreFetch = installFakeFetch({
      offers,
      createResponses: new Map([
        [300, staleResponse(300)],
        [301, staleResponse(301)],
        [302, staleResponse(302)],
      ]),
    });

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, false);
    assert.ok(result.error);
    assert.ok(
      result.error.includes("no_such_ask") || result.error.includes("provision failed"),
      `Expected error about stale offers, got: ${result.error}`,
    );
  });

  it("does not retry on non-retryable errors", async () => {
    const offers = [makeOffer(400), makeOffer(401)];
    const createCalls: number[] = [];

    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (
      input: string | URL | Request,
      init?: RequestInit,
    ): Promise<Response> => {
      const url =
        typeof input === "string" ? input : input instanceof URL ? input.href : input.url;

      if (url.includes("/bundles")) {
        return new Response(JSON.stringify({ offers }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      const askMatch = url.match(/\/asks\/(\d+)\//);
      if (askMatch && init?.method === "PUT") {
        createCalls.push(Number(askMatch[1]));
        return new Response(JSON.stringify({ error: "unauthorized" }), {
          status: 401,
          headers: { "Content-Type": "application/json" },
        });
      }

      return originalFetch(input, init as RequestInit);
    }) as typeof globalThis.fetch;

    restoreFetch = () => {
      globalThis.fetch = originalFetch;
    };

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, false);
    assert.deepEqual(createCalls, [400], "Should only attempt the first offer for non-retryable errors");
  });

  it("fails when no offers found", async () => {
    restoreFetch = installFakeFetch({
      offers: [],
      createResponses: new Map(),
    });

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, false);
    assert.ok(result.error?.includes("No suitable"));
  });
});
