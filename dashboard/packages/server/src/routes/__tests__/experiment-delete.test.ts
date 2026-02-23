import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import Fastify from "fastify";
import cookie from "@fastify/cookie";
import type { FastifyInstance } from "fastify";
import type { ServerConfig } from "../../config.js";
import { Database } from "../../services/database.js";
import { ObjectStore } from "../../services/object-store.js";
import authPlugin from "../../plugins/auth.js";
import experimentsRoutes from "../experiments.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TEST_TOKEN = "test-secret-token";
const JWT_SECRET = "test-jwt-secret-at-least-32-chars-long!";
const AUTH_HEADER = `Bearer ${TEST_TOKEN}`;

const cleanups: Array<() => Promise<void>> = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "exp-delete-route-test-"));
  cleanups.push(async () => {
    await fsp.rm(dir, { recursive: true, force: true });
  });
  return dir;
}

after(async () => {
  for (const fn of cleanups) await fn();
});

async function writeStatusFile(
  experimentsDir: string,
  expId: string,
  status: Record<string, unknown>,
): Promise<void> {
  const metricsDir = path.join(experimentsDir, expId, "dashboard_metrics");
  await fsp.mkdir(metricsDir, { recursive: true });
  await fsp.writeFile(
    path.join(metricsDir, "status.json"),
    JSON.stringify(status),
  );
}

async function buildApp(opts?: {
  redisData?: Map<string, string>;
  redisSets?: Map<string, Set<string>>;
}): Promise<{ app: FastifyInstance; experimentsDir: string; db: Database }> {
  const dir = await freshTmpDir();
  const experimentsDir = path.join(dir, "experiments");
  await fsp.mkdir(experimentsDir, { recursive: true });
  const db = new Database(path.join(dir, "test.db"));

  const kvStore = opts?.redisData ?? new Map<string, string>();
  const setStore = opts?.redisSets ?? new Map<string, Set<string>>();

  const redisMock = {
    get: async (key: string) => kvStore.get(key) ?? null,
    set: async (key: string, value: string) => {
      kvStore.set(key, value);
      return "OK";
    },
    del: async (...keys: string[]) => {
      for (const k of keys) kvStore.delete(k);
      return keys.length;
    },
    srem: async (key: string, member: string) => {
      setStore.get(key)?.delete(member);
      return 1;
    },
    smembers: async (key: string) => Array.from(setStore.get(key) ?? []),
    hmget: async (_hash: string, ...ids: string[]) => ids.map(() => null),
    hgetall: async () => ({}),
    lrange: async () => [],
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
  app.decorate("db", db);
  app.decorate("objectStore", new ObjectStore(null));

  await app.register(cookie);
  await app.register(authPlugin);
  await app.register(experimentsRoutes);

  cleanups.push(async () => {
    db.close();
    await app.close();
  });

  return { app, experimentsDir, db };
}

// ---------------------------------------------------------------------------
// DELETE /api/experiments/:expId
// ---------------------------------------------------------------------------

describe("DELETE /api/experiments/:expId", () => {
  it("returns 401 without auth", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "DELETE",
      url: "/api/experiments/exp-123",
    });

    assert.equal(resp.statusCode, 401);
  });

  it("returns 200 with DeleteResult shape for a completed experiment", async () => {
    const { app, experimentsDir } = await buildApp();

    // Create a completed experiment
    const staleTime = Date.now() / 1000 - 3600;
    await writeStatusFile(experimentsDir, "exp-done", {
      status: "completed",
      last_update: staleTime,
    });
    await fsp.writeFile(
      path.join(experimentsDir, "exp-done", "checkpoint.pth"),
      "fake checkpoint data",
    );

    const resp = await app.inject({
      method: "DELETE",
      url: "/api/experiments/exp-done",
      headers: { authorization: AUTH_HEADER },
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(typeof body.diskDeleted, "boolean");
    assert.equal(typeof body.redisKeysRemoved, "number");
    assert.equal(typeof body.analysesRemoved, "number");
    assert.equal(body.diskDeleted, true);
  });

  it("returns 409 for actively-training experiment", async () => {
    const { app, experimentsDir } = await buildApp();

    const recentTime = Date.now() / 1000 - 60; // 1 minute ago
    await writeStatusFile(experimentsDir, "exp-active", {
      status: "training",
      last_update: recentTime,
    });

    const resp = await app.inject({
      method: "DELETE",
      url: "/api/experiments/exp-active",
      headers: { authorization: AUTH_HEADER },
    });

    assert.equal(resp.statusCode, 409);
    assert.ok(resp.json().error);
  });

  it("returns 200 for already-deleted experiment (idempotent)", async () => {
    const { app } = await buildApp();

    // Experiment doesn't exist at all
    const resp = await app.inject({
      method: "DELETE",
      url: "/api/experiments/nonexistent-exp",
      headers: { authorization: AUTH_HEADER },
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.diskDeleted, false);
  });

  it("GET /api/experiments no longer includes deleted experiment", async () => {
    const { app, experimentsDir } = await buildApp();

    // Create the experiment
    const staleTime = Date.now() / 1000 - 3600;
    await writeStatusFile(experimentsDir, "exp-to-delete", {
      status: "completed",
      last_update: staleTime,
    });

    // Verify it appears in the list
    const listBefore = await app.inject({
      method: "GET",
      url: "/api/experiments",
    });
    const beforeIds = listBefore.json().experiments.map((e: { id: string }) => e.id);
    assert.ok(beforeIds.includes("exp-to-delete"));

    // Delete it
    await app.inject({
      method: "DELETE",
      url: "/api/experiments/exp-to-delete",
      headers: { authorization: AUTH_HEADER },
    });

    // Verify it's gone from the list
    const listAfter = await app.inject({
      method: "GET",
      url: "/api/experiments",
    });
    const afterIds = listAfter.json().experiments.map((e: { id: string }) => e.id);
    assert.equal(afterIds.includes("exp-to-delete"), false);
  });
});
