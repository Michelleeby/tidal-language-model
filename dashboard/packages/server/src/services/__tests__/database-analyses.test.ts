import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { Database } from "../database.js";

// ---------------------------------------------------------------------------
// Temp directory management
// ---------------------------------------------------------------------------

const cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "db-analysis-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

function createDb(dir: string): Database {
  return new Database(path.join(dir, "test.db"));
}

// ---------------------------------------------------------------------------
// createAnalysis
// ---------------------------------------------------------------------------

describe("Database.createAnalysis()", () => {
  it("creates an analysis result with correct fields", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const analysis = db.createAnalysis({
      experimentId: "exp-123",
      analysisType: "trajectory",
      label: "Test trajectory",
      request: { checkpoint: "/path/to/ckpt", prompt: "Once upon a time" },
      data: { text: "generated text", trajectory: { gateSignals: [[0.5]] } },
    });

    assert.ok(analysis.id);
    assert.equal(analysis.experimentId, "exp-123");
    assert.equal(analysis.analysisType, "trajectory");
    assert.equal(analysis.label, "Test trajectory");
    assert.deepEqual(analysis.request, {
      checkpoint: "/path/to/ckpt",
      prompt: "Once upon a time",
    });
    assert.deepEqual(analysis.data, {
      text: "generated text",
      trajectory: { gateSignals: [[0.5]] },
    });
    assert.ok(analysis.sizeBytes > 0);
    assert.ok(analysis.createdAt > 0);

    db.close();
  });

  it("computes sizeBytes from the serialized data JSON", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const smallData = { a: 1 };
    const small = db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "sweep",
      label: "small",
      request: {},
      data: smallData,
    });

    const bigData = { payload: "x".repeat(10000) };
    const big = db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "sweep",
      label: "big",
      request: {},
      data: bigData,
    });

    assert.ok(big.sizeBytes > small.sizeBytes);

    db.close();
  });
});

// ---------------------------------------------------------------------------
// getAnalysis
// ---------------------------------------------------------------------------

describe("Database.getAnalysis()", () => {
  it("returns an analysis by id with parsed JSON", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const created = db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "cross-prompt",
      label: "test",
      request: { prompts: ["a", "b"] },
      data: { batchAnalysis: { perPromptSummaries: {} } },
    });

    const found = db.getAnalysis(created.id);
    assert.ok(found);
    assert.equal(found!.id, created.id);
    assert.deepEqual(found!.request, { prompts: ["a", "b"] });
    assert.deepEqual(found!.data, {
      batchAnalysis: { perPromptSummaries: {} },
    });

    db.close();
  });

  it("returns null for non-existent id", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const found = db.getAnalysis("nonexistent");
    assert.equal(found, null);

    db.close();
  });
});

// ---------------------------------------------------------------------------
// listAnalyses
// ---------------------------------------------------------------------------

describe("Database.listAnalyses()", () => {
  it("returns analyses for a specific experiment", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "trajectory",
      label: "traj-1",
      request: {},
      data: {},
    });
    db.createAnalysis({
      experimentId: "exp-2",
      analysisType: "sweep",
      label: "sweep-2",
      request: {},
      data: {},
    });
    db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "cross-prompt",
      label: "cp-1",
      request: {},
      data: {},
    });

    const results = db.listAnalyses("exp-1");
    assert.equal(results.length, 2);
    // All belong to exp-1
    for (const r of results) {
      assert.equal(r.experimentId, "exp-1");
    }

    db.close();
  });

  it("returns summaries without data field", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "trajectory",
      label: "test",
      request: {},
      data: { big: "payload" },
    });

    const results = db.listAnalyses("exp-1");
    assert.equal(results.length, 1);
    // Summary should NOT have request or data fields
    assert.equal("data" in results[0], false);
    assert.equal("request" in results[0], false);
    // But should have summary fields
    assert.ok(results[0].id);
    assert.equal(results[0].analysisType, "trajectory");
    assert.equal(results[0].label, "test");
    assert.ok(results[0].sizeBytes >= 0);
    assert.ok(results[0].createdAt > 0);

    db.close();
  });

  it("returns results ordered by created_at DESC", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "trajectory",
      label: "first",
      request: {},
      data: {},
    });
    // Ensure different timestamp
    await new Promise((r) => setTimeout(r, 15));
    db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "trajectory",
      label: "second",
      request: {},
      data: {},
    });

    const results = db.listAnalyses("exp-1");
    assert.equal(results[0].label, "second");
    assert.equal(results[1].label, "first");

    db.close();
  });

  it("returns empty array for unknown experiment", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const results = db.listAnalyses("nonexistent");
    assert.deepEqual(results, []);

    db.close();
  });
});

// ---------------------------------------------------------------------------
// deleteAnalysis
// ---------------------------------------------------------------------------

describe("Database.deleteAnalysis()", () => {
  it("deletes an existing analysis", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const created = db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "trajectory",
      label: "to-delete",
      request: {},
      data: {},
    });

    const deleted = db.deleteAnalysis(created.id);
    assert.equal(deleted, true);

    const found = db.getAnalysis(created.id);
    assert.equal(found, null);

    db.close();
  });

  it("returns false for non-existent id", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const deleted = db.deleteAnalysis("nonexistent");
    assert.equal(deleted, false);

    db.close();
  });
});

// ---------------------------------------------------------------------------
// Large payload handling
// ---------------------------------------------------------------------------

describe("Analysis large payload handling", () => {
  it("stores and retrieves large data payloads (50KB+)", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    // Create a ~60KB payload (similar to cross-prompt analysis)
    const largeData: Record<string, unknown> = {};
    for (let i = 0; i < 20; i++) {
      largeData[`prompt_${i}`] = {
        signalStats: { modulation: { mean: Math.random(), std: Math.random() } },
        tokenTexts: Array.from({ length: 50 }, (_, j) => `token_${i}_${j}`),
        gateSignals: Array.from({ length: 50 }, () => [Math.random()]),
      };
    }

    const created = db.createAnalysis({
      experimentId: "exp-large",
      analysisType: "cross-prompt",
      label: "large payload test",
      request: { prompts: Array.from({ length: 20 }, (_, i) => `prompt ${i}`) },
      data: largeData,
    });

    assert.ok(created.sizeBytes > 10000);

    const retrieved = db.getAnalysis(created.id);
    assert.ok(retrieved);
    assert.deepEqual(retrieved!.data, largeData);

    db.close();
  });
});
