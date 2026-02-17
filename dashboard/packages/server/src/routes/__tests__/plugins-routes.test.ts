import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { loadTidalManifest } from "../../services/tidal-manifest-loader.js";
import type { PluginManifest } from "@tidal/shared";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "plugin-route-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

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
    epochCapture: "epoch_(\\\\d+)"
  - phase: rl
    glob: "rl_checkpoint_iter_*.pth"
    epochCapture: "iter_(\\\\d+)"

generation:
  entrypoint: Generator.py
  args:
    config: "--config"
    checkpoint: "--checkpoint"
    prompt: "--prompt"
    maxTokens: "--max_tokens"
    temperature: "--temperature"
    topK: "--top_k"
    rlAgent: "--rl-agent"
    rlCheckpoint: "--rl-checkpoint"
  defaultConfigPath: configs/base_config.yaml
  modes:
    - id: none
      displayName: No Gating
      requiresRLCheckpoint: false
    - id: learned
      displayName: Learned Gating (RL)
      requiresRLCheckpoint: true
  parameters:
    - id: temperature
      displayName: Temperature
      min: 0.1
      max: 2.0
      step: 0.1
      default: 0.8
  modelCheckpointPatterns:
    - "checkpoint_foundational_epoch_*.pth"
  rlCheckpointPatterns:
    - "rl_checkpoint_iter_*.pth"

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
  dockerImage: "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
  requirementsFile: requirements.txt
  gpuTiers:
    standard:
      minGpuRamMb: 16000
      minCpuCores: 16
`;

// ---------------------------------------------------------------------------
// Manifest loading (route-level logic simulation)
// ---------------------------------------------------------------------------

describe("GET /api/plugins (via tidal manifest)", () => {
  it("returns summary for tidal manifest", async () => {
    const tmpDir = await freshTmpDir();
    const manifestPath = path.join(tmpDir, "manifest.yaml");
    await fsp.writeFile(manifestPath, MANIFEST_YAML);

    const manifest = await loadTidalManifest(manifestPath);
    assert.ok(manifest);

    // Simulate route handler logic: build summary
    const plugins = manifest
      ? [
          {
            name: manifest.name,
            displayName: manifest.displayName,
            version: manifest.version,
            trainingPhases: manifest.trainingPhases.map((tp: PluginManifest["trainingPhases"][0]) => ({
              id: tp.id,
              displayName: tp.displayName,
            })),
            generationModes: manifest.generation.modes.map((m: PluginManifest["generation"]["modes"][0]) => ({
              id: m.id,
              displayName: m.displayName,
            })),
          },
        ]
      : [];

    assert.equal(plugins.length, 1);
    assert.equal(plugins[0].name, "tidal");
    assert.equal(plugins[0].trainingPhases.length, 2);
    assert.equal(plugins[0].trainingPhases[0].id, "lm-training");
    assert.equal(plugins[0].trainingPhases[1].id, "rl-training");
    assert.equal(plugins[0].generationModes.length, 2);
    assert.equal(plugins[0].generationModes[0].id, "none");
    assert.equal(plugins[0].generationModes[1].id, "learned");
  });
});

describe("GET /api/plugins/:name (via tidal manifest)", () => {
  it("returns full manifest for tidal", async () => {
    const tmpDir = await freshTmpDir();
    const manifestPath = path.join(tmpDir, "manifest.yaml");
    await fsp.writeFile(manifestPath, MANIFEST_YAML);

    const manifest = await loadTidalManifest(manifestPath);
    assert.ok(manifest);
    assert.equal(manifest.name, "tidal");
    assert.equal(manifest.generation.entrypoint, "Generator.py");
    assert.equal(manifest.infrastructure.dockerImage, "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime");
  });

  it("returns null for missing manifest", async () => {
    const manifest = await loadTidalManifest("/tmp/nonexistent-12345.yaml");
    assert.equal(manifest, null);
  });
});
