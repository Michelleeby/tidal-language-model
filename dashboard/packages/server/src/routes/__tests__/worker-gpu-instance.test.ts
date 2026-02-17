import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import Fastify from "fastify";
import type { FastifyInstance } from "fastify";
import type Redis from "ioredis";
import type { ServerConfig } from "../../config.js";
import type { SSEManager } from "../../services/sse-manager.js";
import workerRoutes from "../workers.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "gpu-inst-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

async function buildApp(experimentsDir: string): Promise<FastifyInstance> {
  const app = Fastify({ logger: false });

  const jobStore = new Map<string, string>();

  const redisMock = {
    hget: async (_hash: string, jobId: string) => jobStore.get(jobId) ?? null,
    hset: async (_hash: string, jobId: string, value: string) => {
      jobStore.set(jobId, value);
      return 0;
    },
    get: async () => null,
    set: async () => "OK",
    del: async () => 0,
    sadd: async () => 0,
    srem: async () => 0,
    expire: async () => 1,
    publish: async () => 0,
    rpush: async () => 0,
    ltrim: async () => "OK",
    lrange: async () => [],
    llen: async () => 0,
    pipeline: () => ({
      exec: async () => [],
      rpush: () => redisMock.pipeline(),
      ltrim: () => redisMock.pipeline(),
      set: () => redisMock.pipeline(),
    }),
    duplicate: () => ({
      subscribe: async () => {},
      on: () => {},
      disconnect: () => {},
    }),
  };

  app.decorate("redis", redisMock as unknown as Redis);
  app.decorate("serverConfig", { experimentsDir } as unknown as ServerConfig);
  app.decorate("sseManager", {
    broadcastJobUpdate: () => {},
    broadcastLogLines: () => {},
  } as unknown as SSEManager);
  app.decorate("tidalManifest", null);
  app.decorate("provisioningChain", { getProvider: () => undefined } as any);
  app.decorate("workerSpawner", {} as any);
  app.decorate("verifyAuth", async () => {});

  await app.register(workerRoutes);

  (app as any).__jobStore = jobStore;
  return app;
}

function seedJob(
  app: FastifyInstance,
  jobId: string,
  overrides: Record<string, unknown> = {},
) {
  const job = {
    jobId,
    type: "lm-training",
    status: "running",
    provider: "vastai",
    config: { type: "lm-training", plugin: "tidal", configPath: "configs/base_config.yaml" },
    createdAt: Date.now(),
    updatedAt: Date.now(),
    ...overrides,
  };
  (app as any).__jobStore.set(jobId, JSON.stringify(job));
}

// ---------------------------------------------------------------------------
// PATCH /api/workers/:jobId/experiment-id — gpu_instance.json persistence
// ---------------------------------------------------------------------------

describe("PATCH /api/workers/:jobId/experiment-id — gpu_instance.json", () => {
  it("writes gpu_instance.json when providerMeta exists", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

    const providerMeta = {
      instanceId: 31562809,
      offerId: 500,
      hostId: 349988,
      gpuName: "RTX A6000",
      numGpus: 1,
      gpuRamMb: 48000,
      costPerHour: 0.65,
      capturedAt: 1700000000000,
    };

    seedJob(app, "job-gpu-1", { providerMeta });

    const resp = await app.inject({
      method: "PATCH",
      url: "/api/workers/job-gpu-1/experiment-id",
      payload: { experimentId: "exp-test-001" },
    });

    assert.equal(resp.statusCode, 200);
    assert.deepEqual(resp.json(), { ok: true, experimentId: "exp-test-001" });

    // Verify gpu_instance.json was written
    const filePath = path.join(tmpDir, "exp-test-001", "gpu_instance.json");
    const content = await fsp.readFile(filePath, "utf-8");
    const parsed = JSON.parse(content);
    assert.equal(parsed.instanceId, 31562809);
    assert.equal(parsed.gpuName, "RTX A6000");
    assert.equal(parsed.costPerHour, 0.65);

    await app.close();
  });

  it("succeeds without creating file when providerMeta is empty", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

    seedJob(app, "job-gpu-2", { providerMeta: undefined });

    const resp = await app.inject({
      method: "PATCH",
      url: "/api/workers/job-gpu-2/experiment-id",
      payload: { experimentId: "exp-test-002" },
    });

    assert.equal(resp.statusCode, 200);
    assert.deepEqual(resp.json(), { ok: true, experimentId: "exp-test-002" });

    // Verify gpu_instance.json was NOT written
    const filePath = path.join(tmpDir, "exp-test-002", "gpu_instance.json");
    await assert.rejects(
      fsp.access(filePath),
      "gpu_instance.json should not exist when providerMeta is empty",
    );

    await app.close();
  });
});
