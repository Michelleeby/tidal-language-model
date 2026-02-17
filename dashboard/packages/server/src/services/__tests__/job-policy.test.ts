import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import type { TrainingJob } from "@tidal/shared";
import {
  ManifestJobPolicy,
  JobPolicyRegistry,
  type JobPolicy,
} from "../job-policy.js";
import { loadTidalManifest } from "../tidal-manifest-loader.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "policy-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

function makeJob(
  overrides: Partial<TrainingJob> & { type: string },
): TrainingJob {
  return {
    jobId: "test-job-" + Math.random().toString(36).slice(2, 8),
    status: "running",
    provider: "local",
    config: {
      type: overrides.type,
      plugin: "tidal",
      configPath: "configs/base_config.yaml",
    },
    createdAt: Date.now(),
    updatedAt: Date.now(),
    ...overrides,
  } as TrainingJob;
}

const MANIFEST_YAML = `
name: tidal
displayName: Tidal Gated Transformer LM
version: 1.0.0
description: A test model

trainingPhases:
  - id: lm-training
    displayName: Language Model Pretraining
    entrypoint: Main.py
    configFiles:
      - configs/base_config.yaml
    args:
      config: "--config"
      resume: "--resume"
    concurrency: 1
    gpuTier: standard
  - id: rl-training
    displayName: RL Gating Controller
    entrypoint: train_rl.py
    configFiles:
      - configs/base_config.yaml
      - configs/rl_config.yaml
    args:
      config: "--config"
      rlConfig: "--rl-config"
      checkpoint: "--checkpoint"
      timesteps: "--timesteps"
    concurrency: 1
    gpuTier: standard

checkpointPatterns:
  - phase: foundational
    glob: "checkpoint_foundational_epoch_*.pth"
generation:
  entrypoint: Generator.py
  args: {}
  defaultConfigPath: configs/base_config.yaml
  modes: []
  parameters: []
  modelCheckpointPatterns: []
  rlCheckpointPatterns: []
metrics:
  redisPrefix: "tidal"
  lm:
    directory: dashboard_metrics
    historyFile: metrics.jsonl
    statusFile: status.json
    latestFile: latest.json
    primaryKeys: []
  rl:
    directory: rl_metrics
    metricsFile: rl_training_metrics.json
    primaryKeys: []
redis:
  jobsHash: "tidal:jobs"
  jobsActiveSet: "tidal:jobs:active"
  signalPrefix: "tidal:job:"
  heartbeatPrefix: "tidal:worker:"
  updatesChannel: "tidal:job:updates"
  experimentsSet: "tidal:experiments"
infrastructure:
  pythonEnv: tidal-env
  dockerImage: "pytorch/pytorch:latest"
  requirementsFile: requirements.txt
  gpuTiers:
    standard:
      minGpuRamMb: 16000
      minCpuCores: 16
`;

// ---------------------------------------------------------------------------
// ManifestJobPolicy
// ---------------------------------------------------------------------------

describe("ManifestJobPolicy", () => {
  it("has the correct type from the manifest phase", () => {
    const policy = new ManifestJobPolicy("lm-training", "LM Training", 1, "standard");
    assert.equal(policy.type, "lm-training");
  });

  it("returns null (no conflict) when no active jobs", () => {
    const policy = new ManifestJobPolicy("lm-training", "LM Training", 1, "standard");
    assert.equal(policy.checkConcurrency([]), null);
  });

  it("returns null when only other-type jobs are active", () => {
    const policy = new ManifestJobPolicy("lm-training", "LM Training", 1, "standard");
    const jobs = [makeJob({ type: "rl-training" })];
    assert.equal(policy.checkConcurrency(jobs), null);
  });

  it("returns error string when same-type job is already active (concurrency=1)", () => {
    const policy = new ManifestJobPolicy("lm-training", "LM Training", 1, "standard");
    const jobs = [makeJob({ type: "lm-training" })];
    const result = policy.checkConcurrency(jobs);
    assert.ok(typeof result === "string");
    assert.ok(result.includes("already running"));
  });

  it("allows multiple jobs when concurrency > 1", () => {
    const policy = new ManifestJobPolicy("lm-training", "LM Training", 3, "standard");
    const jobs = [
      makeJob({ type: "lm-training" }),
      makeJob({ type: "lm-training" }),
    ];
    assert.equal(policy.checkConcurrency(jobs), null);
  });

  it("blocks at concurrency limit", () => {
    const policy = new ManifestJobPolicy("lm-training", "LM Training", 2, "standard");
    const jobs = [
      makeJob({ type: "lm-training" }),
      makeJob({ type: "lm-training" }),
    ];
    const result = policy.checkConcurrency(jobs);
    assert.ok(typeof result === "string");
  });

  it("returns the correct GPU tier", () => {
    const policy = new ManifestJobPolicy("lm-training", "LM Training", 1, "standard");
    assert.equal(policy.gpuTier(), "standard");
  });
});

// ---------------------------------------------------------------------------
// LM + RL coexistence
// ---------------------------------------------------------------------------

describe("LM + RL coexistence", () => {
  it("LM policy allows start when RL job is active", () => {
    const lmPolicy = new ManifestJobPolicy("lm-training", "LM Training", 1, "standard");
    const activeJobs = [makeJob({ type: "rl-training" })];
    assert.equal(lmPolicy.checkConcurrency(activeJobs), null);
  });

  it("RL policy allows start when LM job is active", () => {
    const rlPolicy = new ManifestJobPolicy("rl-training", "RL Training", 1, "standard");
    const activeJobs = [makeJob({ type: "lm-training" })];
    assert.equal(rlPolicy.checkConcurrency(activeJobs), null);
  });

  it("both block same-type concurrent jobs", () => {
    const lm = new ManifestJobPolicy("lm-training", "LM Training", 1, "standard");
    const rl = new ManifestJobPolicy("rl-training", "RL Training", 1, "standard");
    const bothActive = [
      makeJob({ type: "lm-training" }),
      makeJob({ type: "rl-training" }),
    ];
    assert.ok(lm.checkConcurrency(bothActive) !== null);
    assert.ok(rl.checkConcurrency(bothActive) !== null);
  });
});

// ---------------------------------------------------------------------------
// JobPolicyRegistry (manifest-driven)
// ---------------------------------------------------------------------------

describe("JobPolicyRegistry", () => {
  it("builds policies from tidal manifest", async () => {
    const tmpDir = await freshTmpDir();
    const manifestPath = path.join(tmpDir, "manifest.yaml");
    await fsp.writeFile(manifestPath, MANIFEST_YAML);

    const manifest = await loadTidalManifest(manifestPath);
    assert.ok(manifest);

    const registry = new JobPolicyRegistry(manifest);
    const lm = registry.get("lm-training");
    assert.ok(lm);
    assert.equal(lm.type, "lm-training");
  });

  it("returns registered RL policy", async () => {
    const tmpDir = await freshTmpDir();
    const manifestPath = path.join(tmpDir, "manifest.yaml");
    await fsp.writeFile(manifestPath, MANIFEST_YAML);

    const manifest = await loadTidalManifest(manifestPath);
    assert.ok(manifest);

    const registry = new JobPolicyRegistry(manifest);
    const rl = registry.get("rl-training");
    assert.ok(rl);
    assert.equal(rl.type, "rl-training");
  });

  it("returns undefined for unregistered type", async () => {
    const registry = new JobPolicyRegistry(null);
    const result = registry.get("unknown");
    assert.equal(result, undefined);
  });

  it("supports custom policy registration", async () => {
    const registry = new JobPolicyRegistry(null);
    const custom: JobPolicy = {
      type: "lm-training",
      checkConcurrency: () => "blocked",
      gpuTier: () => "standard",
    };
    registry.register(custom);
    assert.equal(registry.get("lm-training"), custom);
  });
});
