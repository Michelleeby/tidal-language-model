import { describe, it, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import type Redis from "ioredis";
import type { MetricPoint, TrainingStatus } from "@tidal/shared";
import { MetricsReader } from "../metrics-reader.js";
import { ExperimentDiscovery } from "../experiment-discovery.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const EXP_ID = "test-exp-001";

function makePoint(step: number): MetricPoint {
  return { step, timestamp: Date.now(), "Losses/Total": 1.5 - step * 0.01 };
}

function makeStatus(status: TrainingStatus["status"] = "training"): TrainingStatus {
  return {
    status,
    last_update: Date.now(),
    current_step: 42,
    total_steps: 100,
  };
}

// ---------------------------------------------------------------------------
// Redis mock factory
// ---------------------------------------------------------------------------

interface MockStore {
  lists: Map<string, string[]>;
  keys: Map<string, string>;
  sets: Map<string, Set<string>>;
}

function createRedisMock(store: MockStore): Redis {
  return {
    lrange: async (key: string, start: number, stop: number) => {
      const list = store.lists.get(key) ?? [];
      // Replicate Redis lrange semantics: negative indices are from the end
      const len = list.length;
      const s = start < 0 ? Math.max(len + start, 0) : start;
      const e = stop < 0 ? len + stop : stop;
      return list.slice(s, e + 1);
    },
    get: async (key: string) => store.keys.get(key) ?? null,
    smembers: async (key: string) => [...(store.sets.get(key) ?? [])],
  } as unknown as Redis;
}

// ---------------------------------------------------------------------------
// Temp directory management
// ---------------------------------------------------------------------------

let tmpDir: string;
let cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "tidal-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// MetricsReader.getFullHistory()
// ---------------------------------------------------------------------------

describe("MetricsReader.getFullHistory()", () => {
  it("returns points from JSONL when file exists and has data", async () => {
    tmpDir = await freshTmpDir();
    const metricsDir = path.join(tmpDir, EXP_ID, "dashboard_metrics");
    await fsp.mkdir(metricsDir, { recursive: true });

    const points = [makePoint(1), makePoint(2), makePoint(3)];
    const jsonl = points.map((p) => JSON.stringify(p)).join("\n") + "\n";
    await fsp.writeFile(path.join(metricsDir, "metrics.jsonl"), jsonl);

    const reader = new MetricsReader(null, tmpDir);
    const result = await reader.getFullHistory(EXP_ID);

    assert.equal(result.length, 3);
    assert.equal(result[0].step, 1);
    assert.equal(result[2].step, 3);
  });

  it("falls back to Redis when JSONL file does not exist", async () => {
    tmpDir = await freshTmpDir();
    // Create experiment dir but no JSONL file
    await fsp.mkdir(path.join(tmpDir, EXP_ID), { recursive: true });

    const redisPoints = [makePoint(10), makePoint(11)];
    const store: MockStore = {
      lists: new Map([
        [
          `tidal:metrics:${EXP_ID}:history`,
          redisPoints.map((p) => JSON.stringify(p)),
        ],
      ]),
      keys: new Map(),
      sets: new Map(),
    };

    const reader = new MetricsReader(createRedisMock(store), tmpDir);
    const result = await reader.getFullHistory(EXP_ID);

    assert.equal(result.length, 2);
    assert.equal(result[0].step, 10);
  });

  it("falls back to Redis when JSONL file exists but is empty", async () => {
    tmpDir = await freshTmpDir();
    const metricsDir = path.join(tmpDir, EXP_ID, "dashboard_metrics");
    await fsp.mkdir(metricsDir, { recursive: true });
    await fsp.writeFile(path.join(metricsDir, "metrics.jsonl"), "");

    const redisPoints = [makePoint(20)];
    const store: MockStore = {
      lists: new Map([
        [
          `tidal:metrics:${EXP_ID}:history`,
          redisPoints.map((p) => JSON.stringify(p)),
        ],
      ]),
      keys: new Map(),
      sets: new Map(),
    };

    const reader = new MetricsReader(createRedisMock(store), tmpDir);
    const result = await reader.getFullHistory(EXP_ID);

    assert.equal(result.length, 1);
    assert.equal(result[0].step, 20);
  });

  it("returns [] when both JSONL and Redis are unavailable", async () => {
    tmpDir = await freshTmpDir();

    const reader = new MetricsReader(null, tmpDir);
    const result = await reader.getFullHistory(EXP_ID);

    assert.deepEqual(result, []);
  });
});

// ---------------------------------------------------------------------------
// MetricsReader.getStatus()
// ---------------------------------------------------------------------------

describe("MetricsReader.getStatus()", () => {
  it("returns status from Redis when available", async () => {
    tmpDir = await freshTmpDir();
    const status = makeStatus("training");

    const store: MockStore = {
      lists: new Map(),
      keys: new Map([[`tidal:status:${EXP_ID}`, JSON.stringify(status)]]),
      sets: new Map(),
    };

    const reader = new MetricsReader(createRedisMock(store), tmpDir);
    const result = await reader.getStatus(EXP_ID);

    assert.ok(result);
    assert.equal(result.status, "training");
    assert.equal(result.current_step, 42);
  });

  it("falls back to disk status.json when Redis returns null", async () => {
    tmpDir = await freshTmpDir();
    const metricsDir = path.join(tmpDir, EXP_ID, "dashboard_metrics");
    await fsp.mkdir(metricsDir, { recursive: true });

    const status = makeStatus("completed");
    await fsp.writeFile(
      path.join(metricsDir, "status.json"),
      JSON.stringify(status),
    );

    // Redis mock returns null for the status key
    const store: MockStore = {
      lists: new Map(),
      keys: new Map(),
      sets: new Map(),
    };

    const reader = new MetricsReader(createRedisMock(store), tmpDir);
    const result = await reader.getStatus(EXP_ID);

    assert.ok(result);
    assert.equal(result.status, "completed");
  });
});

// ---------------------------------------------------------------------------
// ExperimentDiscovery.listExperiments() — Redis fallback for status
// ---------------------------------------------------------------------------

describe("ExperimentDiscovery.listExperiments()", () => {
  it("reads status from Redis when status.json is missing on disk", async () => {
    tmpDir = await freshTmpDir();

    // Create experiment directory with a checkpoint but no status.json
    const expPath = path.join(tmpDir, EXP_ID);
    await fsp.mkdir(expPath, { recursive: true });
    await fsp.writeFile(path.join(expPath, "checkpoint.pth"), "");

    const status = makeStatus("training");
    const store: MockStore = {
      lists: new Map(),
      keys: new Map([[`tidal:status:${EXP_ID}`, JSON.stringify(status)]]),
      sets: new Map([[`tidal:experiments`, new Set([EXP_ID])]]),
    };

    const discovery = new ExperimentDiscovery(createRedisMock(store), tmpDir);
    const experiments = await discovery.listExperiments();

    assert.equal(experiments.length, 1);
    const exp = experiments[0];
    assert.equal(exp.id, EXP_ID);
    assert.ok(exp.status, "status should be populated from Redis");
    assert.equal(exp.status!.status, "training");
    assert.deepEqual(exp.checkpoints, ["checkpoint.pth"]);
  });
});
