import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import Fastify from "fastify";
import cookie from "@fastify/cookie";
import type { FastifyInstance } from "fastify";
import type { ServerConfig } from "../../config.js";
import authPlugin from "../../plugins/auth.js";
import statusRoutes from "../status.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TEST_TOKEN = "test-secret-token";
const JWT_SECRET = "test-jwt-secret-at-least-32-chars-long!";
const AUTH_HEADER = `Bearer ${TEST_TOKEN}`;
const STALENESS_THRESHOLD_S = 300; // 5 minutes

const cleanups: Array<() => Promise<void>> = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "mark-complete-test-"));
  cleanups.push(async () => {
    await fsp.rm(dir, { recursive: true, force: true });
  });
  return dir;
}

after(async () => {
  for (const fn of cleanups) await fn();
});

/**
 * Write a status.json file for an experiment inside the tmp experiments dir.
 */
async function writeStatusFile(
  experimentsDir: string,
  expId: string,
  status: Record<string, unknown>,
): Promise<void> {
  const metricsDir = path.join(experimentsDir, expId, "dashboard_metrics");
  await fsp.mkdir(metricsDir, { recursive: true });
  await fsp.writeFile(
    path.join(metricsDir, "status.json"),
    JSON.stringify(status, null, 2),
  );
}

async function buildApp(opts?: {
  redisJobs?: Map<string, string>;
  redisData?: Map<string, string>;
}): Promise<{ app: FastifyInstance; experimentsDir: string }> {
  const dir = await freshTmpDir();
  const experimentsDir = path.join(dir, "experiments");
  await fsp.mkdir(experimentsDir, { recursive: true });

  const redisData = opts?.redisData ?? new Map<string, string>();
  const redisJobs = opts?.redisJobs ?? new Map<string, string>();

  // Minimal Redis mock (just enough for MetricsReader + JobStore)
  const redisMock = {
    get: async (key: string) => redisData.get(key) ?? null,
    set: async (key: string, value: string) => {
      redisData.set(key, value);
      return "OK";
    },
    smembers: async (key: string) => {
      const raw = redisJobs.get(key);
      return raw ? JSON.parse(raw) : [];
    },
    hmget: async (_hash: string, ...ids: string[]) =>
      ids.map((id) => redisJobs.get(`job:${id}`) ?? null),
    hgetall: async () => ({}),
  };

  const app = Fastify({ logger: false });
  app.decorate("serverConfig", {
    projectRoot: dir,
    experimentsDir,
    authToken: TEST_TOKEN,
    jwtSecret: JWT_SECRET,
  } as unknown as ServerConfig);
  app.decorate("redis", redisMock as any);
  app.decorate("tidalManifest", null);

  await app.register(cookie);
  await app.register(authPlugin);
  await app.register(statusRoutes);

  cleanups.push(async () => {
    await app.close();
  });

  return { app, experimentsDir };
}

// ---------------------------------------------------------------------------
// POST /api/experiments/:expId/status/complete
// ---------------------------------------------------------------------------

describe("POST /api/experiments/:expId/status/complete", () => {
  it("marks a stale experiment as completed", async () => {
    const { app, experimentsDir } = await buildApp();

    // Status file with last_update 10 minutes ago
    const staleTime = Date.now() / 1000 - 600;
    await writeStatusFile(experimentsDir, "exp-stale", {
      status: "training",
      last_update: staleTime,
      current_step: 500,
    });

    const resp = await app.inject({
      method: "POST",
      url: "/api/experiments/exp-stale/status/complete",
      headers: { authorization: AUTH_HEADER },
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.expId, "exp-stale");
    assert.equal(body.status.status, "completed");
    assert.ok(body.status.end_time > 0);

    // Verify the file on disk was updated
    const onDisk = JSON.parse(
      await fsp.readFile(
        path.join(experimentsDir, "exp-stale", "dashboard_metrics", "status.json"),
        "utf-8",
      ),
    );
    assert.equal(onDisk.status, "completed");
  });

  it("returns 409 when experiment was updated recently", async () => {
    const { app, experimentsDir } = await buildApp();

    // Status file with last_update just now
    const recentTime = Date.now() / 1000 - 60; // 1 minute ago
    await writeStatusFile(experimentsDir, "exp-active", {
      status: "training",
      last_update: recentTime,
      current_step: 100,
    });

    const resp = await app.inject({
      method: "POST",
      url: "/api/experiments/exp-active/status/complete",
      headers: { authorization: AUTH_HEADER },
    });

    assert.equal(resp.statusCode, 409);
    const body = resp.json();
    assert.ok(body.error.includes("still be active"));
  });

  it("is idempotent — already-completed experiment returns 200", async () => {
    const { app, experimentsDir } = await buildApp();

    await writeStatusFile(experimentsDir, "exp-done", {
      status: "completed",
      end_time: Date.now() / 1000 - 3600,
      last_update: Date.now() / 1000 - 3600,
    });

    const resp = await app.inject({
      method: "POST",
      url: "/api/experiments/exp-done/status/complete",
      headers: { authorization: AUTH_HEADER },
    });

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.json().status.status, "completed");
  });

  it("returns 401 without auth", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/experiments/exp-123/status/complete",
    });

    assert.equal(resp.statusCode, 401);
  });

  it("returns 404 when experiment has no status file", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/experiments/nonexistent/status/complete",
      headers: { authorization: AUTH_HEADER },
    });

    assert.equal(resp.statusCode, 404);
  });

  it("returns 409 when an active job is linked to the experiment", async () => {
    const redisJobs = new Map<string, string>();
    // Simulate an active job linked to this experiment
    redisJobs.set("tidal:jobs:active", JSON.stringify(["job-1"]));
    redisJobs.set(
      "job:job-1",
      JSON.stringify({
        jobId: "job-1",
        status: "running",
        experimentId: "exp-with-job",
      }),
    );

    const { app, experimentsDir } = await buildApp({ redisJobs });

    const staleTime = Date.now() / 1000 - 600;
    await writeStatusFile(experimentsDir, "exp-with-job", {
      status: "training",
      last_update: staleTime,
      current_step: 200,
    });

    const resp = await app.inject({
      method: "POST",
      url: "/api/experiments/exp-with-job/status/complete",
      headers: { authorization: AUTH_HEADER },
    });

    assert.equal(resp.statusCode, 409);
    const body = resp.json();
    assert.ok(body.error.includes("active job"));
  });
});
