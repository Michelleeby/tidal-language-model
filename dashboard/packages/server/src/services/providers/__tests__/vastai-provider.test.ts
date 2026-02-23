import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import type { TrainingJob } from "@tidal/shared";
import { VastAIProvider, type VastAIProviderConfig, type ProvisionConstraints } from "../vastai-provider.js";

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

// Fast limiter config prevents real waits during tests
const FAST_LIMITER = { capacity: 100, refillRatePerMs: 100 };

function makeProvider(overrides: Partial<VastAIProviderConfig> = {}): VastAIProvider {
  return new VastAIProvider({
    apiKey: "test-key",
    dashboardUrl: "http://localhost:4400",
    authToken: "test-token",
    repoUrl: "https://github.com/test/repo.git",
    log: silentLogger(),
    rateLimiter: FAST_LIMITER,
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
  host_id?: number;
  machine_id?: number;
  gpu_mem_bw?: number;
  total_flops?: number;
  dlperf?: number;
  dlperf_per_dphtotal?: number;
  cpu_name?: string;
  cpu_cores?: number;
  cpu_cores_effective?: number;
  cpu_ram?: number;
  disk_name?: string;
  disk_bw?: number;
  disk_space?: number;
  inet_down?: number;
  inet_up?: number;
  mobo_name?: string;
  cuda_max_good?: number;
  reliability2?: number;
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

function makeRichOffer(id: number, dph = 0.5): FakeOffer {
  return {
    id,
    gpu_name: "RTX A6000",
    gpu_ram: 48000,
    dph_total: dph,
    rentable: true,
    num_gpus: 1,
    host_id: 349988,
    machine_id: 47281,
    gpu_mem_bw: 651.4,
    total_flops: 36.1,
    dlperf: 28.5,
    dlperf_per_dphtotal: 57.0,
    cpu_name: "AMD EPYC 7343 16-Core",
    cpu_cores: 32,
    cpu_cores_effective: 16,
    cpu_ram: 64300,
    disk_name: "SanDisk Extreme 1TB",
    disk_bw: 3673,
    disk_space: 20,
    inet_down: 846.3,
    inet_up: 834.0,
    mobo_name: "H12SSL-i",
    cuda_max_good: 13.0,
    reliability2: 0.99,
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

  it("provision returns rich metadata fields", async () => {
    const offers = [makeRichOffer(500, 0.65)];
    restoreFetch = installFakeFetch({
      offers,
      createResponses: new Map([
        [500, { ok: true, status: 200, body: { new_contract: 31562809 } }],
      ]),
    });

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, true);
    const meta = result.meta!;
    assert.equal(meta.instanceId, 31562809);
    assert.equal(meta.offerId, 500);
    assert.equal(meta.gpuName, "RTX A6000");
    assert.equal(meta.costPerHour, 0.65);
    assert.equal(meta.hostId, 349988);
    assert.equal(meta.machineId, 47281);
    assert.equal(meta.numGpus, 1);
    assert.equal(meta.gpuRamMb, 48000);
    assert.equal(meta.gpuMemBwGbps, 651.4);
    assert.equal(meta.totalFlops, 36.1);
    assert.equal(meta.dlPerf, 28.5);
    assert.equal(meta.dlPerfPerDphTotal, 57.0);
    assert.equal(meta.cpuName, "AMD EPYC 7343 16-Core");
    assert.equal(meta.cpuCores, 32);
    assert.equal(meta.cpuCoresEffective, 16);
    assert.equal(meta.cpuRamMb, 64300);
    assert.equal(meta.diskName, "SanDisk Extreme 1TB");
    assert.equal(meta.diskBwMbps, 3673);
    assert.equal(meta.diskSpaceGb, 20);
    assert.equal(meta.inetDownMbps, 846.3);
    assert.equal(meta.inetUpMbps, 834.0);
    assert.equal(meta.moboName, "H12SSL-i");
    assert.equal(meta.cudaMaxGood, 13.0);
    assert.equal(meta.reliability, 0.99);
    assert.equal(typeof meta.capturedAt, "number");
  });

  it("provision returns graceful nulls for missing optional fields", async () => {
    // makeOffer() only has the minimal fields — all hardware fields absent
    const offers = [makeOffer(600)];
    restoreFetch = installFakeFetch({
      offers,
      createResponses: new Map([
        [600, { ok: true, status: 200, body: { new_contract: 7777 } }],
      ]),
    });

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, true);
    const meta = result.meta!;
    // Core fields present
    assert.equal(meta.instanceId, 7777);
    assert.equal(meta.offerId, 600);
    assert.equal(meta.gpuName, "RTX 4090");
    assert.equal(meta.costPerHour, 0.5);
    // Optional fields gracefully null
    assert.equal(meta.hostId, null);
    assert.equal(meta.machineId, null);
    assert.equal(meta.cpuName, null);
    assert.equal(meta.diskName, null);
    assert.equal(meta.moboName, null);
    assert.equal(meta.gpuMemBwGbps, null);
    assert.equal(meta.dlPerf, null);
    assert.equal(meta.cudaMaxGood, null);
    assert.equal(meta.reliability, null);
    assert.equal(typeof meta.capturedAt, "number");
  });
});

// ---------------------------------------------------------------------------
// Constraint relaxation tiers
// ---------------------------------------------------------------------------

describe("VastAIProvider.provision() — constraint relaxation tiers", () => {
  let restoreFetch: (() => void) | undefined;

  afterEach(() => {
    restoreFetch?.();
    restoreFetch = undefined;
  });

  it("succeeds on tier 0 (strict) without relaxation and sets constraintTier=0", async () => {
    const offers = [makeOffer(700)];
    restoreFetch = installFakeFetch({
      offers,
      createResponses: new Map([[700, { ok: true, status: 200, body: { new_contract: 8000 } }]]),
    });

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, true);
    assert.equal(result.meta?.instanceId, 8000);
    assert.equal(result.meta?.constraintTier, 0, "Should record tier 0 when strict constraints match");
  });

  it("falls back to tier 1 when tier 0 yields no offers", async () => {
    let bundleCallCount = 0;
    const originalFetch = globalThis.fetch;

    globalThis.fetch = (async (
      input: string | URL | Request,
      init?: RequestInit,
    ): Promise<Response> => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;

      if (url.includes("/bundles")) {
        bundleCallCount++;
        // Tier 0: no offers; Tier 1+: one offer
        const offers = bundleCallCount === 1 ? [] : [makeOffer(800)];
        return new Response(JSON.stringify({ offers }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      const askMatch = url.match(/\/asks\/(\d+)\//);
      if (askMatch && init?.method === "PUT") {
        return new Response(JSON.stringify({ new_contract: 9100 }), {
          status: 200,
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

    assert.equal(result.success, true);
    assert.equal(result.meta?.instanceId, 9100);
    assert.equal(result.meta?.constraintTier, 1, "Should record tier 1 when falling back");
    assert.ok(bundleCallCount >= 2, "Should have searched at least twice (once per tier)");
  });

  it("fails after all tiers are exhausted with no offers", async () => {
    restoreFetch = installFakeFetch({
      offers: [], // No offers at any tier
      createResponses: new Map(),
    });

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, false);
    assert.ok(result.error, "Should return an error message");
  });

  it("non-retryable offer error in tier 0 still tries tier 1", async () => {
    let bundleCallCount = 0;
    const createCalls: number[] = [];
    const originalFetch = globalThis.fetch;

    globalThis.fetch = (async (
      input: string | URL | Request,
      init?: RequestInit,
    ): Promise<Response> => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;

      if (url.includes("/bundles")) {
        bundleCallCount++;
        // Both tiers return offers
        return new Response(JSON.stringify({ offers: [makeOffer(900)] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      const askMatch = url.match(/\/asks\/(\d+)\//);
      if (askMatch && init?.method === "PUT") {
        const offerId = Number(askMatch[1]);
        createCalls.push(offerId);
        if (bundleCallCount === 1) {
          // Tier 0: non-retryable auth error
          return new Response(JSON.stringify({ error: "unauthorized" }), {
            status: 401,
            headers: { "Content-Type": "application/json" },
          });
        }
        // Tier 1: success
        return new Response(JSON.stringify({ new_contract: 9200 }), {
          status: 200,
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

    assert.equal(result.success, true, "Should succeed on tier 1 even if tier 0 had non-retryable error");
    assert.equal(result.meta?.constraintTier, 1);
  });

  it("provider meta includes constraintTier on success", async () => {
    const offers = [makeRichOffer(1000, 0.55)];
    restoreFetch = installFakeFetch({
      offers,
      createResponses: new Map([
        [1000, { ok: true, status: 200, body: { new_contract: 9999 } }],
      ]),
    });

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, true);
    assert.equal(typeof result.meta?.constraintTier, "number");
    assert.ok(result.meta!.constraintTier >= 0, "constraintTier should be non-negative");
  });
});

// ---------------------------------------------------------------------------
// 429 rate limit retry
// ---------------------------------------------------------------------------

describe("VastAIProvider.provision() — 429 retry via this.fetch wrapper", () => {
  let restoreFetch: (() => void) | undefined;

  afterEach(() => {
    restoreFetch?.();
    restoreFetch = undefined;
  });

  it("retries on 429 and succeeds after backoff", async () => {
    let searchCallCount = 0;
    let createCallCount = 0;
    const originalFetch = globalThis.fetch;

    globalThis.fetch = (async (
      input: string | URL | Request,
      init?: RequestInit,
    ): Promise<Response> => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;

      if (url.includes("/bundles")) {
        searchCallCount++;
        if (searchCallCount === 1) {
          return new Response("Too Many Requests", {
            status: 429,
            headers: { "Content-Type": "text/plain", "Retry-After": "0" },
          });
        }
        return new Response(JSON.stringify({ offers: [makeOffer(1100)] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      const askMatch = url.match(/\/asks\/(\d+)\//);
      if (askMatch && init?.method === "PUT") {
        createCallCount++;
        return new Response(JSON.stringify({ new_contract: 9300 }), {
          status: 200,
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

    assert.equal(result.success, true);
    assert.equal(result.meta?.instanceId, 9300);
    assert.ok(searchCallCount >= 2, "Should have retried the search after 429");
  });

  it("fails after max 429 retries are exhausted", async () => {
    const originalFetch = globalThis.fetch;

    globalThis.fetch = (async (
      input: string | URL | Request,
    ): Promise<Response> => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;

      if (url.includes("/bundles")) {
        return new Response("Too Many Requests", {
          status: 429,
          headers: { "Content-Type": "text/plain", "Retry-After": "0" },
        });
      }

      return originalFetch(input);
    }) as typeof globalThis.fetch;

    restoreFetch = () => {
      globalThis.fetch = originalFetch;
    };

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, false);
    assert.ok(result.error, "Should return an error after max retries");
  });
});
