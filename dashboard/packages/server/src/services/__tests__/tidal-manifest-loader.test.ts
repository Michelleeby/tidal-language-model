import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { loadTidalManifest } from "../tidal-manifest-loader.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "manifest-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

const VALID_MANIFEST = `
name: tidal
displayName: Tidal Gated Transformer LM
version: 1.0.0
description: A gated transformer language model

trainingPhases:
  - id: lm-training
    displayName: LM Training
    entrypoint: Main.py
    configFiles:
      - configs/base_config.yaml
    args:
      config: "--config"
    concurrency: 1
    gpuTier: standard

checkpointPatterns:
  - phase: foundational
    glob: "checkpoint_foundational_epoch_*.pth"
    epochCapture: "epoch_(\\\\d+)"

generation:
  entrypoint: Generator.py
  args:
    config: "--config"
    checkpoint: "--checkpoint"
    prompt: "--prompt"
    maxTokens: "--max_tokens"
    temperature: "--temperature"
    topK: "--top_k"
  defaultConfigPath: configs/base_config.yaml
  modes:
    - id: none
      displayName: No Gating
      requiresRLCheckpoint: false
  parameters:
    - id: temperature
      displayName: Temperature
      min: 0.1
      max: 2.0
      step: 0.1
      default: 0.8
  modelCheckpointPatterns:
    - "checkpoint_foundational_epoch_*.pth"
  rlCheckpointPatterns: []

metrics:
  redisPrefix: "tidal"
  lm:
    directory: dashboard_metrics
    historyFile: metrics.jsonl
    statusFile: status.json
    latestFile: latest.json
    primaryKeys:
      - "Losses/Total"
  rl:
    directory: rl_metrics
    metricsFile: rl_training_metrics.json
    primaryKeys:
      - episode_rewards

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
      minGpuRamMb: 48000
      minCpuCores: 16
`;

// ---------------------------------------------------------------------------
// loadTidalManifest() — success cases
// ---------------------------------------------------------------------------

describe("loadTidalManifest()", () => {
  it("loads a valid manifest from a file path", async () => {
    const tmpDir = await freshTmpDir();
    const manifestPath = path.join(tmpDir, "manifest.yaml");
    await fsp.writeFile(manifestPath, VALID_MANIFEST);

    const manifest = await loadTidalManifest(manifestPath);
    assert.ok(manifest, "should return a manifest");
    assert.equal(manifest.name, "tidal");
    assert.equal(manifest.displayName, "Tidal Gated Transformer LM");
    assert.equal(manifest.version, "1.0.0");
    assert.equal(manifest.trainingPhases.length, 1);
    assert.equal(manifest.trainingPhases[0].id, "lm-training");
  });

  it("parses all manifest sections correctly", async () => {
    const tmpDir = await freshTmpDir();
    const manifestPath = path.join(tmpDir, "manifest.yaml");
    await fsp.writeFile(manifestPath, VALID_MANIFEST);

    const manifest = await loadTidalManifest(manifestPath);
    assert.ok(manifest);

    // Checkpoint patterns
    assert.equal(manifest.checkpointPatterns.length, 1);
    assert.equal(manifest.checkpointPatterns[0].phase, "foundational");

    // Generation config
    assert.equal(manifest.generation.entrypoint, "Generator.py");
    assert.equal(manifest.generation.modes.length, 1);
    assert.equal(manifest.generation.parameters[0].default, 0.8);

    // Redis config
    assert.equal(manifest.redis.jobsHash, "tidal:jobs");

    // Infrastructure config
    assert.equal(manifest.infrastructure.dockerImage, "pytorch/pytorch:latest");
    assert.equal(manifest.infrastructure.gpuTiers["standard"].minGpuRamMb, 48000);
  });

  // ---------------------------------------------------------------------------
  // loadTidalManifest() — failure cases
  // ---------------------------------------------------------------------------

  it("returns null for a non-existent file", async () => {
    const result = await loadTidalManifest("/tmp/nonexistent-manifest-12345.yaml");
    assert.equal(result, null);
  });

  it("returns null for malformed YAML", async () => {
    const tmpDir = await freshTmpDir();
    const manifestPath = path.join(tmpDir, "manifest.yaml");
    await fsp.writeFile(manifestPath, "name: [unterminated");

    const result = await loadTidalManifest(manifestPath);
    assert.equal(result, null);
  });

  it("returns null when required fields are missing", async () => {
    const tmpDir = await freshTmpDir();
    const manifestPath = path.join(tmpDir, "manifest.yaml");
    await fsp.writeFile(manifestPath, "name: incomplete\nversion: 1.0.0\n");

    const result = await loadTidalManifest(manifestPath);
    assert.equal(result, null);
  });

  it("returns null when trainingPhases is empty", async () => {
    const tmpDir = await freshTmpDir();
    const manifestPath = path.join(tmpDir, "manifest.yaml");
    const badManifest = VALID_MANIFEST.replace(
      /trainingPhases:[\s\S]*?(?=checkpointPatterns)/,
      "trainingPhases: []\n",
    );
    await fsp.writeFile(manifestPath, badManifest);

    const result = await loadTidalManifest(manifestPath);
    assert.equal(result, null);
  });

  // ---------------------------------------------------------------------------
  // Optional logger
  // ---------------------------------------------------------------------------

  it("logs warning on failure when logger is provided", async () => {
    const warnings: string[] = [];
    const logger = {
      info(_msg: string) {},
      warn(msg: string) { warnings.push(msg); },
    };

    await loadTidalManifest("/tmp/nonexistent-12345.yaml", logger);
    assert.ok(warnings.length > 0, "should have logged a warning");
    assert.ok(warnings[0].includes("Failed"), "warning should mention failure");
  });

  it("does not throw when no logger is provided", async () => {
    // Should just return null, no throws
    const result = await loadTidalManifest("/tmp/nonexistent-12345.yaml");
    assert.equal(result, null);
  });
});
