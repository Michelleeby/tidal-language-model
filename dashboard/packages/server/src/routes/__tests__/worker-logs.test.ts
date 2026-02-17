import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import Fastify from "fastify";
import type { FastifyInstance } from "fastify";
import type Redis from "ioredis";
import type { ServerConfig } from "../../config.js";
import type { SSEManager } from "../../services/sse-manager.js";
import workerRoutes from "../workers.js";
import jobRoutes from "../jobs.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const AUTH_TOKEN = "test-token-abc";

/**
 * Build a minimal Fastify instance with enough decoration to run
 * worker + job routes for log streaming tests.
 */
async function buildApp(): Promise<FastifyInstance> {
  const app = Fastify({ logger: false });

  // In-memory Redis mock with list operations for logs
  const jobStore = new Map<string, string>();
  const lists = new Map<string, string[]>();

  const redisMock = {
    hget: async (_hash: string, jobId: string) => jobStore.get(jobId) ?? null,
    hset: async () => 0,
    get: async () => null,
    set: async () => "OK",
    del: async () => 0,
    sadd: async () => 0,
    srem: async () => 0,
    rpush: async (key: string, ...values: string[]) => {
      if (!lists.has(key)) lists.set(key, []);
      lists.get(key)!.push(...values);
      return lists.get(key)!.length;
    },
    ltrim: async (key: string, start: number, stop: number) => {
      const list = lists.get(key);
      if (!list) return "OK";
      // Handle negative indices
      const len = list.length;
      const s = start < 0 ? Math.max(0, len + start) : start;
      const e = stop < 0 ? len + stop : stop;
      lists.set(key, list.slice(s, e + 1));
      return "OK";
    },
    lrange: async (key: string, start: number, stop: number) => {
      const list = lists.get(key) ?? [];
      const len = list.length;
      const s = start < 0 ? Math.max(0, len + start) : start;
      const e = stop < 0 ? len + stop : stop;
      return list.slice(s, e + 1);
    },
    llen: async (key: string) => {
      return lists.get(key)?.length ?? 0;
    },
    expire: async () => 1,
    publish: async () => 0,
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
  app.decorate("serverConfig", {
    experimentsDir: "/tmp/tidal-test-experiments",
  } as unknown as ServerConfig);
  app.decorate("sseManager", {
    broadcastJobUpdate: () => {},
    broadcastLogLines: () => {},
  } as unknown as SSEManager);
  app.decorate("tidalManifest", null);
  app.decorate("provisioningChain", { getProvider: () => undefined } as any);
  app.decorate("workerSpawner", {} as any);
  app.decorate("verifyAuth", async () => {});

  await app.register(workerRoutes);

  // Expose for seeding test data
  (app as any).__jobStore = jobStore;
  (app as any).__lists = lists;
  return app;
}

/**
 * Build a separate app instance for job routes (GET /api/jobs/:jobId/logs).
 */
async function buildJobApp(): Promise<FastifyInstance> {
  const app = Fastify({ logger: false });

  const jobStore = new Map<string, string>();
  const lists = new Map<string, string[]>();

  const redisMock = {
    hget: async (_hash: string, jobId: string) => jobStore.get(jobId) ?? null,
    hset: async () => 0,
    get: async () => null,
    set: async () => "OK",
    del: async () => 0,
    sadd: async () => 0,
    srem: async () => 0,
    lrange: async (key: string, start: number, stop: number) => {
      const list = lists.get(key) ?? [];
      const len = list.length;
      const s = start < 0 ? Math.max(0, len + start) : start;
      const e = stop < 0 ? len + stop : stop;
      return list.slice(s, e + 1);
    },
    llen: async (key: string) => {
      return lists.get(key)?.length ?? 0;
    },
    publish: async () => 0,
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
  app.decorate("serverConfig", {
    experimentsDir: "/tmp/tidal-test-experiments",
    defaultComputeProvider: "local",
  } as unknown as ServerConfig);
  app.decorate("sseManager", {
    broadcastJobUpdate: () => {},
    broadcastLogLines: () => {},
  } as unknown as SSEManager);
  app.decorate("tidalManifest", null);
  app.decorate("provisioningChain", { getProvider: () => undefined } as any);
  app.decorate("workerSpawner", { cleanup: () => {} } as any);
  app.decorate("verifyAuth", async () => {});

  await app.register(jobRoutes);

  (app as any).__jobStore = jobStore;
  (app as any).__lists = lists;
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
    provider: "local",
    config: { type: "lm-training", plugin: "tidal", configPath: "configs/base_config.yaml" },
    createdAt: Date.now(),
    updatedAt: Date.now(),
    ...overrides,
  };
  (app as any).__jobStore.set(jobId, JSON.stringify(job));
}

// ---------------------------------------------------------------------------
// POST /api/workers/:jobId/logs
// ---------------------------------------------------------------------------

describe("POST /api/workers/:jobId/logs", () => {
  it("ingests log lines to Redis list", async () => {
    const app = await buildApp();
    seedJob(app, "job-1");

    const lines = [
      { timestamp: 1000, stream: "stdout", line: "Training started" },
      { timestamp: 1001, stream: "stderr", line: "Warning: low memory" },
    ];

    const resp = await app.inject({
      method: "POST",
      url: "/api/workers/job-1/logs",
      payload: { lines },
    });

    assert.equal(resp.statusCode, 200);
    assert.deepEqual(resp.json(), { ok: true, ingested: 2 });

    // Check Redis list
    const redis = (app as any).redis;
    const stored = await redis.lrange("tidal:logs:job-1", 0, -1);
    assert.equal(stored.length, 2);

    const first = JSON.parse(stored[0]);
    assert.equal(first.line, "Training started");
    assert.equal(first.stream, "stdout");

    await app.close();
  });

  it("handles empty lines array", async () => {
    const app = await buildApp();
    seedJob(app, "job-1");

    const resp = await app.inject({
      method: "POST",
      url: "/api/workers/job-1/logs",
      payload: { lines: [] },
    });

    assert.equal(resp.statusCode, 200);
    assert.deepEqual(resp.json(), { ok: true, ingested: 0 });

    await app.close();
  });

  it("returns 400 when lines field is missing", async () => {
    const app = await buildApp();
    seedJob(app, "job-1");

    const resp = await app.inject({
      method: "POST",
      url: "/api/workers/job-1/logs",
      payload: {},
    });

    assert.equal(resp.statusCode, 400);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// GET /api/jobs/:jobId/logs
// ---------------------------------------------------------------------------

describe("GET /api/jobs/:jobId/logs", () => {
  it("retrieves stored log lines", async () => {
    const app = await buildJobApp();
    seedJob(app, "job-2");

    // Seed some log lines directly
    const lists = (app as any).__lists;
    const lines = [
      JSON.stringify({ timestamp: 100, stream: "stdout", line: "line 1" }),
      JSON.stringify({ timestamp: 101, stream: "stderr", line: "line 2" }),
      JSON.stringify({ timestamp: 102, stream: "stdout", line: "line 3" }),
    ];
    lists.set("tidal:logs:job-2", lines);

    const resp = await app.inject({
      method: "GET",
      url: "/api/jobs/job-2/logs",
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.jobId, "job-2");
    assert.equal(body.lines.length, 3);
    assert.equal(body.totalLines, 3);
    assert.equal(body.lines[0].line, "line 1");

    await app.close();
  });

  it("supports offset and limit pagination", async () => {
    const app = await buildJobApp();
    seedJob(app, "job-3");

    const lists = (app as any).__lists;
    const lines = Array.from({ length: 10 }, (_, i) =>
      JSON.stringify({ timestamp: i, stream: "stdout", line: `line ${i}` }),
    );
    lists.set("tidal:logs:job-3", lines);

    const resp = await app.inject({
      method: "GET",
      url: "/api/jobs/job-3/logs?offset=3&limit=2",
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.lines.length, 2);
    assert.equal(body.lines[0].line, "line 3");
    assert.equal(body.lines[1].line, "line 4");
    assert.equal(body.totalLines, 10);

    await app.close();
  });

  it("returns empty lines for job with no logs", async () => {
    const app = await buildJobApp();
    seedJob(app, "job-4");

    const resp = await app.inject({
      method: "GET",
      url: "/api/jobs/job-4/logs",
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.jobId, "job-4");
    assert.equal(body.lines.length, 0);
    assert.equal(body.totalLines, 0);

    await app.close();
  });
});
