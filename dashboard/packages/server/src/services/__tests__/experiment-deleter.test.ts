import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { ExperimentDeleter } from "../experiment-deleter.js";
import { Database } from "../database.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const cleanups: Array<() => Promise<void>> = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "exp-deleter-test-"));
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

async function writeArchiveManifest(
  experimentsDir: string,
  expId: string,
  state: "uploading" | "complete" | "failed",
): Promise<void> {
  const expDir = path.join(experimentsDir, expId);
  await fsp.mkdir(expDir, { recursive: true });
  await fsp.writeFile(
    path.join(expDir, "_archive_manifest.json"),
    JSON.stringify({ state, archivedAt: Date.now(), spacesPrefix: `experiments/${expId}/` }),
  );
}

function makeMockRedis(data?: Map<string, string>, sets?: Map<string, Set<string>>) {
  const kvStore = data ?? new Map<string, string>();
  const setStore = sets ?? new Map<string, Set<string>>();
  const deletedKeys: string[] = [];

  return {
    get: async (key: string) => kvStore.get(key) ?? null,
    del: async (...keys: string[]) => {
      for (const k of keys) {
        kvStore.delete(k);
        deletedKeys.push(k);
      }
      return keys.length;
    },
    srem: async (key: string, member: string) => {
      const s = setStore.get(key);
      if (s) s.delete(member);
      return 1;
    },
    smembers: async (key: string) => {
      return Array.from(setStore.get(key) ?? []);
    },
    hgetall: async () => ({}),
    hmget: async (_hash: string, ..._ids: string[]) => [],
    lrange: async () => [],
    _deletedKeys: deletedKeys,
    _kvStore: kvStore,
  };
}

// ---------------------------------------------------------------------------
// canDelete() tests
// ---------------------------------------------------------------------------

describe("ExperimentDeleter.canDelete()", () => {
  it("returns ok for completed experiment", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const db = new Database(path.join(dir, "test.db"));

    const staleTime = Date.now() / 1000 - 3600; // 1 hour ago
    await writeStatusFile(experimentsDir, "exp-done", {
      status: "completed",
      last_update: staleTime,
    });

    const deleter = new ExperimentDeleter(null, experimentsDir, db);
    const result = await deleter.canDelete("exp-done");
    assert.equal(result.ok, true);

    db.close();
  });

  it("returns rejection for actively-training experiment", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const db = new Database(path.join(dir, "test.db"));

    const recentTime = Date.now() / 1000 - 60; // 1 minute ago
    await writeStatusFile(experimentsDir, "exp-active", {
      status: "training",
      last_update: recentTime,
    });

    const deleter = new ExperimentDeleter(null, experimentsDir, db);
    const result = await deleter.canDelete("exp-active");
    assert.equal(result.ok, false);
    assert.ok(result.reason?.includes("active") || result.reason?.includes("training"));

    db.close();
  });

  it("returns rejection when active job is linked", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const db = new Database(path.join(dir, "test.db"));

    // Stale status (training > 5min ago)
    const staleTime = Date.now() / 1000 - 600;
    await writeStatusFile(experimentsDir, "exp-with-job", {
      status: "training",
      last_update: staleTime,
    });

    // Mock Redis with active job for this experiment
    const setStore = new Map<string, Set<string>>();
    setStore.set("tidal:jobs:active", new Set(["job-1"]));
    const kvStore = new Map<string, string>();
    kvStore.set(
      "tidal:jobs",
      JSON.stringify({
        "job-1": JSON.stringify({ jobId: "job-1", status: "running", experimentId: "exp-with-job" }),
      }),
    );

    const redis = {
      ...makeMockRedis(kvStore, setStore),
      hmget: async (_hash: string, ...ids: string[]) => {
        return ids.map((id) => {
          // Simulate the hash lookup
          const raw = `{"jobId":"${id}","status":"running","experimentId":"exp-with-job"}`;
          return raw;
        });
      },
    };

    const deleter = new ExperimentDeleter(redis as any, experimentsDir, db);
    const result = await deleter.canDelete("exp-with-job");
    assert.equal(result.ok, false);
    assert.ok(result.reason?.includes("job"));

    db.close();
  });

  it("returns rejection when archival is in progress", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const db = new Database(path.join(dir, "test.db"));

    await writeArchiveManifest(experimentsDir, "exp-archiving", "uploading");
    // Also give it a completed status so that doesn't block
    const staleTime = Date.now() / 1000 - 3600;
    await writeStatusFile(experimentsDir, "exp-archiving", {
      status: "completed",
      last_update: staleTime,
    });

    const deleter = new ExperimentDeleter(null, experimentsDir, db);
    const result = await deleter.canDelete("exp-archiving");
    assert.equal(result.ok, false);
    assert.ok(result.reason?.includes("archiv"));

    db.close();
  });

  it("returns ok for non-existent experiment (idempotent)", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const db = new Database(path.join(dir, "test.db"));

    const deleter = new ExperimentDeleter(null, experimentsDir, db);
    const result = await deleter.canDelete("nonexistent-exp");
    assert.equal(result.ok, true);

    db.close();
  });
});

// ---------------------------------------------------------------------------
// delete() tests
// ---------------------------------------------------------------------------

describe("ExperimentDeleter.delete()", () => {
  it("removes experiment directory from disk", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const db = new Database(path.join(dir, "test.db"));

    // Create experiment directory
    const expDir = path.join(experimentsDir, "exp-123");
    await fsp.mkdir(expDir, { recursive: true });
    await fsp.writeFile(path.join(expDir, "metadata.json"), "{}");

    const deleter = new ExperimentDeleter(null, experimentsDir, db);
    const result = await deleter.delete("exp-123");
    assert.equal(result.diskDeleted, true);

    // Verify directory is gone
    const exists = await fsp.access(expDir).then(() => true).catch(() => false);
    assert.equal(exists, false);

    db.close();
  });

  it("removes Redis keys", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const db = new Database(path.join(dir, "test.db"));

    const kvStore = new Map<string, string>();
    kvStore.set("tidal:status:exp-456", '{"status":"completed"}');
    kvStore.set("tidal:metrics:exp-456:latest", '{"step":100}');
    const setStore = new Map<string, Set<string>>();
    setStore.set("tidal:experiments", new Set(["exp-456"]));

    const redis = makeMockRedis(kvStore, setStore);

    const deleter = new ExperimentDeleter(redis as any, experimentsDir, db);
    const result = await deleter.delete("exp-456");
    assert.ok(result.redisKeysRemoved >= 1);

    // Verify keys are gone
    assert.equal(kvStore.get("tidal:status:exp-456"), undefined);
    assert.equal(kvStore.get("tidal:metrics:exp-456:latest"), undefined);

    db.close();
  });

  it("removes analysis_results from SQLite", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const db = new Database(path.join(dir, "test.db"));

    // Create some analyses in the DB
    db.createAnalysis({
      experimentId: "exp-789",
      analysisType: "trajectory",
      label: "test",
      request: {},
      data: {},
    });
    db.createAnalysis({
      experimentId: "exp-789",
      analysisType: "sweep",
      label: "test2",
      request: {},
      data: {},
    });

    const analyses = db.listAnalyses("exp-789");
    assert.equal(analyses.length, 2);

    const deleter = new ExperimentDeleter(null, experimentsDir, db);
    const result = await deleter.delete("exp-789");
    assert.equal(result.analysesRemoved, 2);

    // Verify analyses are gone
    const remaining = db.listAnalyses("exp-789");
    assert.equal(remaining.length, 0);

    db.close();
  });

  it("is idempotent (second call doesn't error)", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const db = new Database(path.join(dir, "test.db"));

    const expDir = path.join(experimentsDir, "exp-idem");
    await fsp.mkdir(expDir, { recursive: true });

    const deleter = new ExperimentDeleter(null, experimentsDir, db);

    await deleter.delete("exp-idem");
    // Second call — directory already gone, should not throw
    const result = await deleter.delete("exp-idem");
    assert.equal(result.diskDeleted, false);

    db.close();
  });

  it("handles missing directory gracefully", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const db = new Database(path.join(dir, "test.db"));

    const deleter = new ExperimentDeleter(null, experimentsDir, db);
    const result = await deleter.delete("no-such-experiment");
    assert.equal(result.diskDeleted, false);
    assert.equal(result.redisKeysRemoved, 0);
    assert.equal(result.analysesRemoved, 0);

    db.close();
  });

  it("handles Redis unavailable gracefully (still deletes disk + SQLite)", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const db = new Database(path.join(dir, "test.db"));

    const expDir = path.join(experimentsDir, "exp-noredis");
    await fsp.mkdir(expDir, { recursive: true });
    await fsp.writeFile(path.join(expDir, "metadata.json"), "{}");

    db.createAnalysis({
      experimentId: "exp-noredis",
      analysisType: "trajectory",
      label: "test",
      request: {},
      data: {},
    });

    // No Redis
    const deleter = new ExperimentDeleter(null, experimentsDir, db);
    const result = await deleter.delete("exp-noredis");

    assert.equal(result.diskDeleted, true);
    assert.equal(result.redisKeysRemoved, 0);
    assert.equal(result.analysesRemoved, 1);

    // Verify disk is deleted
    const exists = await fsp.access(expDir).then(() => true).catch(() => false);
    assert.equal(exists, false);

    db.close();
  });
});
