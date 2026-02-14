import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import Fastify from "fastify";
import type { FastifyInstance } from "fastify";
import type { ServerConfig } from "../../config.js";
import { PluginRegistry } from "../../services/plugin-registry.js";
import checkpointsRoutes from "../checkpoints.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "ckpt-route-test-"));
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
description: A gated transformer language model

trainingPhases:
  - id: lm-training
    displayName: Language Model Pretraining
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
  - phase: rl
    glob: "rl_checkpoint_iter_*.pth"
    epochCapture: "iter_(\\\\d+)"
  - phase: final
    glob: "*_v*.pth"
    excludePrefix: "rl_"

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
      minGpuRamMb: 16000
      minCpuCores: 16
`;

/**
 * Build a Fastify app wired with a real PluginRegistry and checkpoints route.
 */
async function buildApp(
  experimentsDir: string,
  pluginsDir: string,
): Promise<FastifyInstance> {
  const app = Fastify({ logger: false });

  app.decorate("serverConfig", { experimentsDir } as unknown as ServerConfig);

  const registry = new PluginRegistry(pluginsDir);
  await registry.load();
  app.decorate("pluginRegistry", registry);

  await app.register(checkpointsRoutes);
  return app;
}

// ---------------------------------------------------------------------------
// Integration: manifest → PluginRegistry → checkpoints route → response
// ---------------------------------------------------------------------------

describe("GET /api/experiments/:expId/checkpoints — phase classification", () => {
  it("classifies foundational checkpoints when plugin is loaded", async () => {
    const tmpDir = await freshTmpDir();

    // Set up plugin directory with manifest
    const pluginsDir = path.join(tmpDir, "plugins");
    const pluginDir = path.join(pluginsDir, "tidal");
    await fsp.mkdir(pluginDir, { recursive: true });
    await fsp.writeFile(path.join(pluginDir, "manifest.yaml"), MANIFEST_YAML);

    // Set up experiment directory with checkpoint files
    const experimentsDir = path.join(tmpDir, "experiments");
    const expDir = path.join(experimentsDir, "exp-001");
    await fsp.mkdir(expDir, { recursive: true });
    await fsp.writeFile(path.join(expDir, "checkpoint_foundational_epoch_1.pth"), "fake");
    await fsp.writeFile(path.join(expDir, "checkpoint_foundational_epoch_5.pth"), "fake");

    const app = await buildApp(experimentsDir, pluginsDir);
    const resp = await app.inject({
      method: "GET",
      url: "/api/experiments/exp-001/checkpoints",
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.expId, "exp-001");
    assert.equal(body.checkpoints.length, 2);

    const cp1 = body.checkpoints.find(
      (c: any) => c.filename === "checkpoint_foundational_epoch_1.pth",
    );
    assert.ok(cp1, "should find epoch 1 checkpoint");
    assert.equal(cp1.phase, "foundational");
    assert.equal(cp1.epoch, 1);

    const cp5 = body.checkpoints.find(
      (c: any) => c.filename === "checkpoint_foundational_epoch_5.pth",
    );
    assert.ok(cp5, "should find epoch 5 checkpoint");
    assert.equal(cp5.phase, "foundational");
    assert.equal(cp5.epoch, 5);

    await app.close();
  });

  it("classifies RL and final checkpoints correctly", async () => {
    const tmpDir = await freshTmpDir();

    const pluginsDir = path.join(tmpDir, "plugins");
    const pluginDir = path.join(pluginsDir, "tidal");
    await fsp.mkdir(pluginDir, { recursive: true });
    await fsp.writeFile(path.join(pluginDir, "manifest.yaml"), MANIFEST_YAML);

    const experimentsDir = path.join(tmpDir, "experiments");
    const expDir = path.join(experimentsDir, "exp-002");
    await fsp.mkdir(expDir, { recursive: true });
    await fsp.writeFile(path.join(expDir, "rl_checkpoint_iter_500.pth"), "fake");
    await fsp.writeFile(path.join(expDir, "transformer-lm_v1.0.0.pth"), "fake");

    const app = await buildApp(experimentsDir, pluginsDir);
    const resp = await app.inject({
      method: "GET",
      url: "/api/experiments/exp-002/checkpoints",
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();

    const rlCp = body.checkpoints.find(
      (c: any) => c.filename === "rl_checkpoint_iter_500.pth",
    );
    assert.ok(rlCp, "should find RL checkpoint");
    assert.equal(rlCp.phase, "rl");
    assert.equal(rlCp.epoch, 500);

    const finalCp = body.checkpoints.find(
      (c: any) => c.filename === "transformer-lm_v1.0.0.pth",
    );
    assert.ok(finalCp, "should find final checkpoint");
    assert.equal(finalCp.phase, "final");

    await app.close();
  });

  it("returns 'unknown' phase when no plugin is loaded", async () => {
    const tmpDir = await freshTmpDir();

    // Empty plugins dir — no manifest to load
    const pluginsDir = path.join(tmpDir, "plugins");
    await fsp.mkdir(pluginsDir, { recursive: true });

    const experimentsDir = path.join(tmpDir, "experiments");
    const expDir = path.join(experimentsDir, "exp-003");
    await fsp.mkdir(expDir, { recursive: true });
    await fsp.writeFile(path.join(expDir, "checkpoint_foundational_epoch_1.pth"), "fake");

    const app = await buildApp(experimentsDir, pluginsDir);
    const resp = await app.inject({
      method: "GET",
      url: "/api/experiments/exp-003/checkpoints",
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.checkpoints.length, 1);
    assert.equal(body.checkpoints[0].phase, "unknown");

    await app.close();
  });

  it("returns empty checkpoints for non-existent experiment", async () => {
    const tmpDir = await freshTmpDir();

    const pluginsDir = path.join(tmpDir, "plugins");
    await fsp.mkdir(pluginsDir, { recursive: true });

    const experimentsDir = path.join(tmpDir, "experiments");
    await fsp.mkdir(experimentsDir, { recursive: true });

    const app = await buildApp(experimentsDir, pluginsDir);
    const resp = await app.inject({
      method: "GET",
      url: "/api/experiments/nonexistent/checkpoints",
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.expId, "nonexistent");
    assert.equal(body.checkpoints.length, 0);

    await app.close();
  });

  it("ignores non-.pth files in experiment directory", async () => {
    const tmpDir = await freshTmpDir();

    const pluginsDir = path.join(tmpDir, "plugins");
    const pluginDir = path.join(pluginsDir, "tidal");
    await fsp.mkdir(pluginDir, { recursive: true });
    await fsp.writeFile(path.join(pluginDir, "manifest.yaml"), MANIFEST_YAML);

    const experimentsDir = path.join(tmpDir, "experiments");
    const expDir = path.join(experimentsDir, "exp-004");
    await fsp.mkdir(expDir, { recursive: true });
    await fsp.writeFile(path.join(expDir, "checkpoint_foundational_epoch_2.pth"), "fake");
    await fsp.writeFile(path.join(expDir, "status.json"), "{}");
    await fsp.writeFile(path.join(expDir, "metrics.jsonl"), "");

    const app = await buildApp(experimentsDir, pluginsDir);
    const resp = await app.inject({
      method: "GET",
      url: "/api/experiments/exp-004/checkpoints",
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.checkpoints.length, 1);
    assert.equal(body.checkpoints[0].filename, "checkpoint_foundational_epoch_2.pth");
    assert.equal(body.checkpoints[0].phase, "foundational");

    await app.close();
  });
});
