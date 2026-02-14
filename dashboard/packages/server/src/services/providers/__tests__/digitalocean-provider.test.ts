import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import type { TrainingJob } from "@tidal/shared";
import {
  DigitalOceanProvider,
  type DigitalOceanProviderConfig,
} from "../digitalocean-provider.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeJob(overrides: Partial<TrainingJob> = {}): TrainingJob {
  return {
    jobId: "job-" + Math.random().toString(36).slice(2, 8),
    type: "lm-training",
    status: "provisioning",
    provider: "digitalocean",
    config: {
      type: "lm-training",
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
  } as unknown as DigitalOceanProviderConfig["log"];
}

function makeProvider(
  overrides: Partial<DigitalOceanProviderConfig> = {},
): DigitalOceanProvider {
  return new DigitalOceanProvider({
    apiKey: "test-key",
    dashboardUrl: "http://localhost:4400",
    authToken: "test-token",
    repoUrl: "https://github.com/test/repo.git",
    log: silentLogger(),
    region: "tor1",
    ...overrides,
  });
}

interface FakeDOSize {
  slug: string;
  memory: number;
  vcpus: number;
  disk: number;
  price_monthly: number;
  price_hourly: number;
  regions: string[];
  available: boolean;
  description: string;
  gpu_info?: {
    count: number;
    vram: { amount: number; unit: string };
    model: string;
  };
}

function makeDOSize(overrides: Partial<FakeDOSize> = {}): FakeDOSize {
  return {
    slug: "gpu-h100x1-80gb",
    memory: 262144,
    vcpus: 20,
    disk: 500,
    price_monthly: 2160,
    price_hourly: 2.95,
    regions: ["tor1", "nyc1"],
    available: true,
    description: "GPU H100 x1 80GB",
    gpu_info: {
      count: 1,
      vram: { amount: 80, unit: "gib" },
      model: "nvidia_h100",
    },
    ...overrides,
  };
}

/**
 * Install a fake `fetch` that routes DigitalOcean API calls to handlers.
 */
function installFakeFetch(opts: {
  sizes?: FakeDOSize[];
  createResponse?: { ok: boolean; status: number; body: unknown };
  deleteResponse?: { status: number };
  getDropletResponse?: { status: number; body?: unknown };
}) {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = (async (
    input: string | URL | Request,
    init?: RequestInit,
  ): Promise<Response> => {
    const url =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;

    // List sizes endpoint
    if (url.includes("/v2/sizes")) {
      return new Response(
        JSON.stringify({ sizes: opts.sizes ?? [] }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }

    // Create droplet endpoint
    if (url.includes("/v2/droplets") && init?.method === "POST") {
      const resp = opts.createResponse ?? {
        ok: true,
        status: 202,
        body: { droplet: { id: 12345 } },
      };
      return new Response(JSON.stringify(resp.body), {
        status: resp.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Delete droplet endpoint
    const deleteMatch = url.match(/\/v2\/droplets\/(\d+)$/);
    if (deleteMatch && init?.method === "DELETE") {
      const resp = opts.deleteResponse ?? { status: 204 };
      return new Response(null, { status: resp.status });
    }

    // Get droplet endpoint
    const getMatch = url.match(/\/v2\/droplets\/(\d+)$/);
    if (getMatch && (!init?.method || init.method === "GET")) {
      const resp = opts.getDropletResponse ?? {
        status: 200,
        body: { droplet: { id: Number(getMatch[1]), status: "active" } },
      };
      if (resp.status === 404) {
        return new Response("not found", { status: 404 });
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
// Tests — canProvision
// ---------------------------------------------------------------------------

describe("DigitalOceanProvider.canProvision()", () => {
  it("returns true with API key", async () => {
    const provider = makeProvider({ apiKey: "dop_v1_test123" });
    assert.equal(await provider.canProvision(), true);
  });

  it("returns false without API key", async () => {
    const provider = makeProvider({ apiKey: null });
    assert.equal(await provider.canProvision(), false);
  });

  it("returns false with empty API key", async () => {
    const provider = makeProvider({ apiKey: "" });
    assert.equal(await provider.canProvision(), false);
  });
});

// ---------------------------------------------------------------------------
// Tests — provision
// ---------------------------------------------------------------------------

describe("DigitalOceanProvider.provision()", () => {
  let restoreFetch: (() => void) | undefined;

  afterEach(() => {
    restoreFetch?.();
    restoreFetch = undefined;
  });

  it("succeeds — queries sizes, creates droplet", async () => {
    const size = makeDOSize({
      slug: "gpu-h100x1-80gb",
      price_hourly: 2.95,
    });

    restoreFetch = installFakeFetch({
      sizes: [size],
      createResponse: {
        ok: true,
        status: 202,
        body: { droplet: { id: 99001 } },
      },
    });

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, true);
    assert.equal(result.meta?.dropletId, 99001);
    assert.equal(result.meta?.sizeSlug, "gpu-h100x1-80gb");
    assert.equal(result.meta?.pricePerHour, 2.95);
  });

  it("filters sizes by GPU RAM from tier spec", async () => {
    const smallGpu = makeDOSize({
      slug: "gpu-a10-24gb",
      price_hourly: 1.5,
      gpu_info: {
        count: 1,
        vram: { amount: 24, unit: "gib" },
        model: "nvidia_a10",
      },
    });
    const bigGpu = makeDOSize({
      slug: "gpu-h100x1-80gb",
      price_hourly: 2.95,
      gpu_info: {
        count: 1,
        vram: { amount: 80, unit: "gib" },
        model: "nvidia_h100",
      },
    });

    restoreFetch = installFakeFetch({
      sizes: [smallGpu, bigGpu],
      createResponse: {
        ok: true,
        status: 202,
        body: { droplet: { id: 99002 } },
      },
    });

    // Provider with gpuTiers that require 48GB+
    const provider = makeProvider({
      gpuTiers: {
        standard: { minGpuRamMb: 48000, minCpuCores: 16 },
      },
    });
    const result = await provider.provision(makeJob());

    assert.equal(result.success, true);
    // Should pick the 80GB H100, not the 24GB A10
    assert.equal(result.meta?.sizeSlug, "gpu-h100x1-80gb");
  });

  it("picks cheapest matching size", async () => {
    const expensive = makeDOSize({
      slug: "gpu-h100x8-640gb",
      price_hourly: 20.0,
      vcpus: 160,
    });
    const cheap = makeDOSize({
      slug: "gpu-h100x1-80gb",
      price_hourly: 2.95,
    });

    restoreFetch = installFakeFetch({
      sizes: [expensive, cheap],
      createResponse: {
        ok: true,
        status: 202,
        body: { droplet: { id: 99003 } },
      },
    });

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, true);
    assert.equal(result.meta?.sizeSlug, "gpu-h100x1-80gb");
    assert.equal(result.meta?.pricePerHour, 2.95);
  });

  it("fails when no GPU sizes match", async () => {
    const cpuOnly = makeDOSize({
      slug: "s-4vcpu-8gb",
      gpu_info: undefined,
    });

    restoreFetch = installFakeFetch({ sizes: [cpuOnly] });

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, false);
    assert.ok(result.error?.includes("No suitable"));
  });

  it("fails when droplet creation returns error", async () => {
    restoreFetch = installFakeFetch({
      sizes: [makeDOSize()],
      createResponse: {
        ok: false,
        status: 422,
        body: { id: "unprocessable_entity", message: "region unavailable" },
      },
    });

    const provider = makeProvider();
    const result = await provider.provision(makeJob());

    assert.equal(result.success, false);
    assert.ok(result.error);
  });

  it("fails when missing dashboardUrl or authToken", async () => {
    restoreFetch = installFakeFetch({ sizes: [makeDOSize()] });

    const provider = makeProvider({ dashboardUrl: null });
    const result = await provider.provision(makeJob());

    assert.equal(result.success, false);
    assert.ok(result.error?.includes("Missing"));
  });
});

// ---------------------------------------------------------------------------
// Tests — deprovision
// ---------------------------------------------------------------------------

describe("DigitalOceanProvider.deprovision()", () => {
  let restoreFetch: (() => void) | undefined;

  afterEach(() => {
    restoreFetch?.();
    restoreFetch = undefined;
  });

  it("calls DELETE on droplet", async () => {
    let deletedId: string | undefined;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (
      input: string | URL | Request,
      init?: RequestInit,
    ): Promise<Response> => {
      const url =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.href
            : input.url;
      const match = url.match(/\/v2\/droplets\/(\d+)$/);
      if (match && init?.method === "DELETE") {
        deletedId = match[1];
        return new Response(null, { status: 204 });
      }
      return originalFetch(input, init as RequestInit);
    }) as typeof globalThis.fetch;
    restoreFetch = () => {
      globalThis.fetch = originalFetch;
    };

    const provider = makeProvider();
    const job = makeJob({ providerMeta: { dropletId: 55555 } });

    await provider.deprovision(job);
    assert.equal(deletedId, "55555");
  });

  it("handles 404 gracefully", async () => {
    restoreFetch = installFakeFetch({
      deleteResponse: { status: 404 },
    });

    const provider = makeProvider();
    const job = makeJob({ providerMeta: { dropletId: 99999 } });

    // Should not throw
    await provider.deprovision(job);
  });
});

// ---------------------------------------------------------------------------
// Tests — isAlive
// ---------------------------------------------------------------------------

describe("DigitalOceanProvider.isAlive()", () => {
  let restoreFetch: (() => void) | undefined;

  afterEach(() => {
    restoreFetch?.();
    restoreFetch = undefined;
  });

  it("returns true for 'active' status", async () => {
    restoreFetch = installFakeFetch({
      getDropletResponse: {
        status: 200,
        body: { droplet: { id: 1, status: "active" } },
      },
    });

    const provider = makeProvider();
    const job = makeJob({ providerMeta: { dropletId: 1 } });
    assert.equal(await provider.isAlive(job), true);
  });

  it("returns true for 'new' status", async () => {
    restoreFetch = installFakeFetch({
      getDropletResponse: {
        status: 200,
        body: { droplet: { id: 1, status: "new" } },
      },
    });

    const provider = makeProvider();
    const job = makeJob({ providerMeta: { dropletId: 1 } });
    assert.equal(await provider.isAlive(job), true);
  });

  it("returns false for 'off' status", async () => {
    restoreFetch = installFakeFetch({
      getDropletResponse: {
        status: 200,
        body: { droplet: { id: 1, status: "off" } },
      },
    });

    const provider = makeProvider();
    const job = makeJob({ providerMeta: { dropletId: 1 } });
    assert.equal(await provider.isAlive(job), false);
  });

  it("returns false for 'archive' status", async () => {
    restoreFetch = installFakeFetch({
      getDropletResponse: {
        status: 200,
        body: { droplet: { id: 1, status: "archive" } },
      },
    });

    const provider = makeProvider();
    const job = makeJob({ providerMeta: { dropletId: 1 } });
    assert.equal(await provider.isAlive(job), false);
  });

  it("returns false on 404", async () => {
    restoreFetch = installFakeFetch({
      getDropletResponse: { status: 404 },
    });

    const provider = makeProvider();
    const job = makeJob({ providerMeta: { dropletId: 1 } });
    assert.equal(await provider.isAlive(job), false);
  });
});
