import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import * as fsp from "node:fs/promises";
import * as path from "node:path";
import * as os from "node:os";
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
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "tidal-ckpt-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

const AUTH_TOKEN = "test-token-abc";

/**
 * Build a minimal Fastify instance with just enough decoration to run
 * the worker routes (GET checkpoint download endpoint).
 */
async function buildApp(experimentsDir: string): Promise<FastifyInstance> {
  const app = Fastify({ logger: false });

  // Minimal Redis mock — only needs hget for JobStore.get()
  const jobStore = new Map<string, string>();
  const redisMock = {
    hget: async (_hash: string, jobId: string) => jobStore.get(jobId) ?? null,
    hset: async () => 0,
    get: async () => null,
    set: async () => "OK",
    del: async () => 0,
    sadd: async () => 0,
    srem: async () => 0,
    pipeline: () => ({
      exec: async () => [],
      rpush: () => redisMock.pipeline(),
      ltrim: () => redisMock.pipeline(),
      set: () => redisMock.pipeline(),
    }),
    publish: async () => 0,
  };
  app.decorate("redis", redisMock as unknown as Redis);
  app.decorate("serverConfig", { experimentsDir } as unknown as ServerConfig);
  app.decorate("sseManager", { broadcastJobUpdate: () => {} } as unknown as SSEManager);
  app.decorate("verifyAuth", async () => {});

  await app.register(workerRoutes);
  // Expose for seeding test data
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
    type: "rl-training",
    status: "starting",
    provider: "vastai",
    config: { type: "rl-training", configPath: "configs/base_config.yaml" },
    createdAt: Date.now(),
    updatedAt: Date.now(),
    ...overrides,
  };
  (app as any).__jobStore.set(jobId, JSON.stringify(job));
}

// ---------------------------------------------------------------------------
// Filename regex validation
// ---------------------------------------------------------------------------

describe("GET /api/workers/:jobId/checkpoints/:filename — filename validation", () => {
  it("rejects filenames without .pth or .json extension", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);
    seedJob(app, "job-1");

    const resp = await app.inject({
      method: "GET",
      url: "/api/workers/job-1/checkpoints/model.bin",
    });
    assert.equal(resp.statusCode, 400);
    assert.match(resp.json().error, /Invalid filename/);

    await app.close();
  });

  it("accepts valid .json filenames", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);
    seedJob(app, "job-1", { experimentId: "exp-1" });

    // File doesn't exist → 404, but filename validation passed
    const resp = await app.inject({
      method: "GET",
      url: "/api/workers/job-1/checkpoints/ablation_results.json",
    });
    assert.equal(resp.statusCode, 404);
    assert.match(resp.json().error, /Checkpoint not found/);

    await app.close();
  });

  it("rejects path traversal attempts", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);
    seedJob(app, "job-1");

    const resp = await app.inject({
      method: "GET",
      url: "/api/workers/job-1/checkpoints/..%2F..%2Fetc%2Fpasswd.pth",
    });
    // Encoded slashes in the filename should fail the regex
    assert.equal(resp.statusCode, 400);

    await app.close();
  });

  it("accepts valid .pth filenames", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);
    seedJob(app, "job-1", { experimentId: "exp-1" });

    // File doesn't exist → 404, but filename validation passed
    const resp = await app.inject({
      method: "GET",
      url: "/api/workers/job-1/checkpoints/transformer-lm_v1.0.0.pth",
    });
    assert.equal(resp.statusCode, 404);
    assert.match(resp.json().error, /Checkpoint not found/);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// File streaming
// ---------------------------------------------------------------------------

describe("GET /api/workers/:jobId/checkpoints/:filename — file streaming", () => {
  it("streams file with correct headers", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);
    seedJob(app, "job-1", { experimentId: "exp-abc" });

    // Create a fake checkpoint file
    const expDir = path.join(tmpDir, "exp-abc");
    await fsp.mkdir(expDir, { recursive: true });
    const content = Buffer.from("fake model weights 12345");
    await fsp.writeFile(path.join(expDir, "model.pth"), content);

    const resp = await app.inject({
      method: "GET",
      url: "/api/workers/job-1/checkpoints/model.pth",
    });

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.headers["content-type"], "application/octet-stream");
    assert.equal(
      resp.headers["content-disposition"],
      'attachment; filename="model.pth"',
    );
    assert.equal(Number(resp.headers["content-length"]), content.length);
    assert.deepEqual(resp.rawPayload, content);

    await app.close();
  });

  it("resolves expId from query param when job has no experimentId", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);
    seedJob(app, "job-2"); // No experimentId on job

    const expDir = path.join(tmpDir, "exp-fallback");
    await fsp.mkdir(expDir, { recursive: true });
    const content = Buffer.from("weights");
    await fsp.writeFile(path.join(expDir, "checkpoint.pth"), content);

    const resp = await app.inject({
      method: "GET",
      url: "/api/workers/job-2/checkpoints/checkpoint.pth?expId=exp-fallback",
    });

    assert.equal(resp.statusCode, 200);
    assert.deepEqual(resp.rawPayload, content);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// 404 cases
// ---------------------------------------------------------------------------

describe("GET /api/workers/:jobId/checkpoints/:filename — 404 cases", () => {
  it("returns 404 when checkpoint file does not exist", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);
    seedJob(app, "job-1", { experimentId: "exp-1" });

    const resp = await app.inject({
      method: "GET",
      url: "/api/workers/job-1/checkpoints/missing.pth",
    });

    assert.equal(resp.statusCode, 404);
    assert.match(resp.json().error, /Checkpoint not found/);

    await app.close();
  });

  it("returns 404 when job does not exist", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

    const resp = await app.inject({
      method: "GET",
      url: "/api/workers/nonexistent/checkpoints/model.pth",
    });

    assert.equal(resp.statusCode, 404);
    assert.match(resp.json().error, /Job not found/);

    await app.close();
  });

  it("returns 400 when no experimentId available", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);
    seedJob(app, "job-1"); // No experimentId, no query param

    const resp = await app.inject({
      method: "GET",
      url: "/api/workers/job-1/checkpoints/model.pth",
    });

    assert.equal(resp.statusCode, 400);
    assert.match(resp.json().error, /experimentId/);

    await app.close();
  });
});
