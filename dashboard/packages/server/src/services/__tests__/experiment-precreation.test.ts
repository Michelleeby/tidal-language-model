import { describe, it, after, before } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { preCreateExperiment } from "../experiment-precreation.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function freshTmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "precreate-test-"));
}

/** Minimal Redis mock that records SADD calls. */
function mockRedis() {
  const saddCalls: [string, string][] = [];
  return {
    sadd: async (key: string, member: string) => {
      saddCalls.push([key, member]);
      return 1;
    },
    saddCalls,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("preCreateExperiment", () => {
  let tmpDir: string;

  before(() => {
    tmpDir = freshTmpDir();
  });

  after(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("generates an experiment ID with the correct format for LM jobs", async () => {
    const redis = mockRedis();
    const experimentsDir = path.join(tmpDir, "exp-lm-format");
    fs.mkdirSync(experimentsDir, { recursive: true });

    const id = await preCreateExperiment(
      "lm-training",
      { configPath: "plugins/tidal/configs/base_config.yaml" },
      experimentsDir,
      redis as any,
      tmpDir,
    );

    // Format: YYYYMMDD-HHmmss-commit_{hash}-config_{hex}
    assert.match(id, /^\d{8}-\d{6}-commit_\w+-config_[0-9a-f]{10}$/);
  });

  it("generates an experiment ID with the correct format for RL jobs", async () => {
    const redis = mockRedis();
    const experimentsDir = path.join(tmpDir, "exp-rl-format");
    fs.mkdirSync(experimentsDir, { recursive: true });

    const id = await preCreateExperiment(
      "rl-training",
      {
        configPath: "plugins/tidal/configs/base_config.yaml",
        checkpoint: "experiments/some-lm-exp/transformer-lm_v1.0.0.pth",
      },
      experimentsDir,
      redis as any,
      tmpDir,
    );

    // Format: YYYYMMDD-HHmmss-commit_{hash}-rl_{hex}
    assert.match(id, /^\d{8}-\d{6}-commit_\w+-rl_[0-9a-f]{10}$/);
  });

  it("creates the experiment directory on disk", async () => {
    const redis = mockRedis();
    const experimentsDir = path.join(tmpDir, "exp-dir-create");
    fs.mkdirSync(experimentsDir, { recursive: true });

    const id = await preCreateExperiment(
      "lm-training",
      { configPath: "plugins/tidal/configs/base_config.yaml" },
      experimentsDir,
      redis as any,
      tmpDir,
    );

    const expDir = path.join(experimentsDir, id);
    assert.ok(fs.existsSync(expDir), `Expected directory to exist: ${expDir}`);
    assert.ok(fs.statSync(expDir).isDirectory());
  });

  it("writes correct metadata.json for LM jobs", async () => {
    const redis = mockRedis();
    const experimentsDir = path.join(tmpDir, "exp-meta-lm");
    fs.mkdirSync(experimentsDir, { recursive: true });

    const id = await preCreateExperiment(
      "lm-training",
      { configPath: "plugins/tidal/configs/base_config.yaml" },
      experimentsDir,
      redis as any,
      tmpDir,
    );

    const metaPath = path.join(experimentsDir, id, "metadata.json");
    assert.ok(fs.existsSync(metaPath));
    const meta = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
    assert.equal(meta.type, "lm");
    assert.ok(meta.created_at);
    assert.equal(meta.source_experiment_id, null);
    assert.equal(meta.source_checkpoint, null);
  });

  it("writes correct metadata.json for RL jobs with source experiment", async () => {
    const redis = mockRedis();
    const experimentsDir = path.join(tmpDir, "exp-meta-rl");
    fs.mkdirSync(experimentsDir, { recursive: true });

    const checkpointPath = "experiments/20250101-commit_abc-config_def/transformer-lm_v1.0.0.pth";
    const id = await preCreateExperiment(
      "rl-training",
      {
        configPath: "plugins/tidal/configs/base_config.yaml",
        checkpoint: checkpointPath,
      },
      experimentsDir,
      redis as any,
      tmpDir,
    );

    const metaPath = path.join(experimentsDir, id, "metadata.json");
    const meta = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
    assert.equal(meta.type, "rl");
    assert.equal(meta.source_experiment_id, "20250101-commit_abc-config_def");
    assert.equal(meta.source_checkpoint, checkpointPath);
  });

  it("registers experiment in Redis via SADD", async () => {
    const redis = mockRedis();
    const experimentsDir = path.join(tmpDir, "exp-redis");
    fs.mkdirSync(experimentsDir, { recursive: true });

    const id = await preCreateExperiment(
      "lm-training",
      { configPath: "plugins/tidal/configs/base_config.yaml" },
      experimentsDir,
      redis as any,
      tmpDir,
    );

    assert.equal(redis.saddCalls.length, 1);
    assert.equal(redis.saddCalls[0][0], "tidal:experiments");
    assert.equal(redis.saddCalls[0][1], id);
  });

  it("for resume jobs, returns existing experiment ID without creating directory", async () => {
    const redis = mockRedis();
    const experimentsDir = path.join(tmpDir, "exp-resume");
    fs.mkdirSync(experimentsDir, { recursive: true });

    // Pre-create the experiment directory to simulate an existing experiment
    const existingExpId = "20250101-120000-commit_abc-config_def1234567";
    const existingDir = path.join(experimentsDir, existingExpId);
    fs.mkdirSync(existingDir, { recursive: true });

    const resumeExpDir = `experiments/${existingExpId}`;

    const id = await preCreateExperiment(
      "lm-training",
      {
        configPath: "plugins/tidal/configs/base_config.yaml",
        resumeExpDir,
      },
      experimentsDir,
      redis as any,
      tmpDir,
    );

    assert.equal(id, existingExpId);
    // Should still register in Redis (so SSE picks it up)
    assert.equal(redis.saddCalls.length, 1);
  });

  it("derives source_experiment_id from checkpoint path for RL jobs", async () => {
    const redis = mockRedis();
    const experimentsDir = path.join(tmpDir, "exp-derive");
    fs.mkdirSync(experimentsDir, { recursive: true });

    const id = await preCreateExperiment(
      "rl-training",
      {
        configPath: "plugins/tidal/configs/base_config.yaml",
        checkpoint: "experiments/my-lm-experiment/model.pth",
      },
      experimentsDir,
      redis as any,
      tmpDir,
    );

    const meta = JSON.parse(
      fs.readFileSync(path.join(experimentsDir, id, "metadata.json"), "utf-8"),
    );
    assert.equal(meta.source_experiment_id, "my-lm-experiment");
  });
});
