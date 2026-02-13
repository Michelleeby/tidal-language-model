import { describe, it, after, before } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { PluginRegistry } from "../../services/plugin-registry.js";

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
// Plugin registry integration (route-level logic)
// ---------------------------------------------------------------------------

describe("GET /api/plugins (via registry)", () => {
  it("lists all plugins with summary info", async () => {
    const tmpDir = await freshTmpDir();
    const pluginDir = path.join(tmpDir, "tidal");
    await fsp.mkdir(pluginDir, { recursive: true });
    await fsp.writeFile(path.join(pluginDir, "manifest.yaml"), MANIFEST_YAML);

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    // Simulate route handler logic: build summary list
    const plugins = registry.list().map((p) => ({
      name: p.name,
      displayName: p.displayName,
      version: p.version,
      trainingPhases: p.trainingPhases.map((tp) => ({
        id: tp.id,
        displayName: tp.displayName,
      })),
      generationModes: p.generation.modes.map((m) => ({
        id: m.id,
        displayName: m.displayName,
      })),
    }));

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

describe("GET /api/plugins/:name (via registry)", () => {
  it("returns full manifest for known plugin", async () => {
    const tmpDir = await freshTmpDir();
    const pluginDir = path.join(tmpDir, "tidal");
    await fsp.mkdir(pluginDir, { recursive: true });
    await fsp.writeFile(path.join(pluginDir, "manifest.yaml"), MANIFEST_YAML);

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    const plugin = registry.get("tidal");
    assert.ok(plugin);
    assert.equal(plugin.name, "tidal");
    assert.equal(plugin.generation.entrypoint, "Generator.py");
    assert.equal(plugin.infrastructure.dockerImage, "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime");
  });

  it("returns undefined for unknown plugin", async () => {
    const tmpDir = await freshTmpDir();
    const pluginDir = path.join(tmpDir, "tidal");
    await fsp.mkdir(pluginDir, { recursive: true });
    await fsp.writeFile(path.join(pluginDir, "manifest.yaml"), MANIFEST_YAML);

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    assert.equal(registry.get("nonexistent"), undefined);
  });
});
