import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { PluginRegistry } from "../plugin-registry.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "plugin-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

const VALID_MANIFEST = `
name: test-model
displayName: Test Model
version: 1.0.0
description: A test model plugin

trainingPhases:
  - id: lm-training
    displayName: LM Training
    entrypoint: train.py
    configFiles:
      - config.yaml
    args:
      config: "--config"
    concurrency: 1
    gpuTier: standard

checkpointPatterns:
  - phase: foundational
    glob: "checkpoint_epoch_*.pth"
    epochCapture: "epoch_(\\\\d+)"

generation:
  entrypoint: generate.py
  args:
    config: "--config"
    checkpoint: "--checkpoint"
    prompt: "--prompt"
    maxTokens: "--max_tokens"
    temperature: "--temperature"
    topK: "--top_k"
  defaultConfigPath: config.yaml
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
    - "checkpoint_epoch_*.pth"
  rlCheckpointPatterns: []

metrics:
  redisPrefix: "test"
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
  jobsHash: "test:jobs"
  jobsActiveSet: "test:jobs:active"
  signalPrefix: "test:job:"
  heartbeatPrefix: "test:worker:"
  updatesChannel: "test:job:updates"
  experimentsSet: "test:experiments"

infrastructure:
  pythonEnv: test-env
  dockerImage: "pytorch/pytorch:latest"
  requirementsFile: requirements.txt
  gpuTiers:
    standard:
      minGpuRamMb: 16000
      minCpuCores: 16
`;

async function createPluginDir(
  baseDir: string,
  name: string,
  yaml: string,
): Promise<void> {
  const dir = path.join(baseDir, name);
  await fsp.mkdir(dir, { recursive: true });
  await fsp.writeFile(path.join(dir, "manifest.yaml"), yaml);
}

// ---------------------------------------------------------------------------
// PluginRegistry.load()
// ---------------------------------------------------------------------------

describe("PluginRegistry.load()", () => {
  it("loads a valid manifest from the plugins directory", async () => {
    const tmpDir = await freshTmpDir();
    await createPluginDir(tmpDir, "test-model", VALID_MANIFEST);

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    const plugin = registry.get("test-model");
    assert.ok(plugin, "should find the plugin");
    assert.equal(plugin.name, "test-model");
    assert.equal(plugin.displayName, "Test Model");
    assert.equal(plugin.version, "1.0.0");
    assert.equal(plugin.trainingPhases.length, 1);
    assert.equal(plugin.trainingPhases[0].id, "lm-training");
  });

  it("loads multiple plugins from the plugins directory", async () => {
    const tmpDir = await freshTmpDir();
    await createPluginDir(tmpDir, "model-a", VALID_MANIFEST.replace(/test-model/g, "model-a").replace(/Test Model/g, "Model A"));
    await createPluginDir(tmpDir, "model-b", VALID_MANIFEST.replace(/test-model/g, "model-b").replace(/Test Model/g, "Model B"));

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    const list = registry.list();
    assert.equal(list.length, 2);
    assert.ok(registry.get("model-a"));
    assert.ok(registry.get("model-b"));
  });

  it("skips directories without manifest.yaml", async () => {
    const tmpDir = await freshTmpDir();
    await createPluginDir(tmpDir, "valid-plugin", VALID_MANIFEST);
    // Create a directory without manifest
    await fsp.mkdir(path.join(tmpDir, "no-manifest"), { recursive: true });

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    assert.equal(registry.list().length, 1);
    assert.ok(registry.get("valid-plugin") === undefined);
    assert.ok(registry.get("test-model"));
  });

  it("rejects manifest missing required fields", async () => {
    const tmpDir = await freshTmpDir();
    const invalidManifest = `
name: bad-plugin
version: 1.0.0
`;
    await createPluginDir(tmpDir, "bad-plugin", invalidManifest);

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    // Bad plugin should be skipped, not crash
    assert.equal(registry.list().length, 0);
    assert.equal(registry.get("bad-plugin"), undefined);
  });

  it("handles empty plugins directory", async () => {
    const tmpDir = await freshTmpDir();

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    assert.equal(registry.list().length, 0);
  });

  it("handles non-existent plugins directory", async () => {
    const registry = new PluginRegistry("/tmp/nonexistent-dir-12345");
    await registry.load();

    assert.equal(registry.list().length, 0);
  });
});

// ---------------------------------------------------------------------------
// PluginRegistry.get()
// ---------------------------------------------------------------------------

describe("PluginRegistry.get()", () => {
  it("returns undefined for unknown plugin name", async () => {
    const tmpDir = await freshTmpDir();
    await createPluginDir(tmpDir, "test-model", VALID_MANIFEST);

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    assert.equal(registry.get("nonexistent"), undefined);
  });
});

// ---------------------------------------------------------------------------
// PluginRegistry.getDefault()
// ---------------------------------------------------------------------------

describe("PluginRegistry.getDefault()", () => {
  it("returns the first loaded plugin when only one exists", async () => {
    const tmpDir = await freshTmpDir();
    await createPluginDir(tmpDir, "test-model", VALID_MANIFEST);

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    const defaultPlugin = registry.getDefault();
    assert.ok(defaultPlugin);
    assert.equal(defaultPlugin.name, "test-model");
  });

  it("returns undefined when no plugins are loaded", async () => {
    const tmpDir = await freshTmpDir();
    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    assert.equal(registry.getDefault(), undefined);
  });
});

// ---------------------------------------------------------------------------
// PluginRegistry.getPhase()
// ---------------------------------------------------------------------------

describe("PluginRegistry.getPhase()", () => {
  it("returns the phase by ID from a plugin", async () => {
    const tmpDir = await freshTmpDir();
    await createPluginDir(tmpDir, "test-model", VALID_MANIFEST);

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    const phase = registry.getPhase("test-model", "lm-training");
    assert.ok(phase);
    assert.equal(phase.id, "lm-training");
    assert.equal(phase.entrypoint, "train.py");
  });

  it("returns undefined for unknown phase ID", async () => {
    const tmpDir = await freshTmpDir();
    await createPluginDir(tmpDir, "test-model", VALID_MANIFEST);

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    assert.equal(registry.getPhase("test-model", "unknown-phase"), undefined);
  });

  it("returns undefined for unknown plugin name", async () => {
    const tmpDir = await freshTmpDir();
    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    assert.equal(registry.getPhase("nonexistent", "lm-training"), undefined);
  });
});

// ---------------------------------------------------------------------------
// Manifest field validation
// ---------------------------------------------------------------------------

describe("Manifest field parsing", () => {
  it("parses checkpoint patterns correctly", async () => {
    const tmpDir = await freshTmpDir();
    await createPluginDir(tmpDir, "test-model", VALID_MANIFEST);

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    const plugin = registry.get("test-model")!;
    assert.equal(plugin.checkpointPatterns.length, 1);
    assert.equal(plugin.checkpointPatterns[0].phase, "foundational");
    assert.equal(plugin.checkpointPatterns[0].glob, "checkpoint_epoch_*.pth");
  });

  it("parses generation config correctly", async () => {
    const tmpDir = await freshTmpDir();
    await createPluginDir(tmpDir, "test-model", VALID_MANIFEST);

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    const plugin = registry.get("test-model")!;
    assert.equal(plugin.generation.entrypoint, "generate.py");
    assert.equal(plugin.generation.modes.length, 1);
    assert.equal(plugin.generation.parameters.length, 1);
    assert.equal(plugin.generation.parameters[0].id, "temperature");
    assert.equal(plugin.generation.parameters[0].default, 0.8);
  });

  it("parses redis config correctly", async () => {
    const tmpDir = await freshTmpDir();
    await createPluginDir(tmpDir, "test-model", VALID_MANIFEST);

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    const plugin = registry.get("test-model")!;
    assert.equal(plugin.redis.jobsHash, "test:jobs");
    assert.equal(plugin.redis.updatesChannel, "test:job:updates");
  });

  it("parses infrastructure config correctly", async () => {
    const tmpDir = await freshTmpDir();
    await createPluginDir(tmpDir, "test-model", VALID_MANIFEST);

    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    const plugin = registry.get("test-model")!;
    assert.equal(plugin.infrastructure.dockerImage, "pytorch/pytorch:latest");
    assert.ok(plugin.infrastructure.gpuTiers["standard"]);
    assert.equal(plugin.infrastructure.gpuTiers["standard"].minGpuRamMb, 16000);
  });
});

// ---------------------------------------------------------------------------
// Diagnostic logging
// ---------------------------------------------------------------------------

function mockLogger() {
  const calls: Array<{ level: "info" | "warn"; message: string }> = [];
  return {
    calls,
    info(msg: string) {
      calls.push({ level: "info", message: msg });
    },
    warn(msg: string) {
      calls.push({ level: "warn", message: msg });
    },
  };
}

describe("PluginRegistry diagnostic logging", () => {
  it("logs successful plugin load and summary", async () => {
    const tmpDir = await freshTmpDir();
    await createPluginDir(tmpDir, "test-model", VALID_MANIFEST);

    const logger = mockLogger();
    const registry = new PluginRegistry(tmpDir, logger);
    await registry.load();

    const loadMsg = logger.calls.find(
      (c) => c.level === "info" && c.message.includes("Loaded plugin: test-model"),
    );
    assert.ok(loadMsg, "should log info when a plugin loads successfully");

    const summaryMsg = logger.calls.find(
      (c) => c.level === "info" && c.message.includes("1 plugin(s) loaded"),
    );
    assert.ok(summaryMsg, "should log summary after loading");
  });

  it("warns when plugins directory does not exist", async () => {
    const logger = mockLogger();
    const registry = new PluginRegistry("/tmp/nonexistent-dir-99999", logger);
    await registry.load();

    const warnMsg = logger.calls.find(
      (c) => c.level === "warn" && c.message.includes("not found"),
    );
    assert.ok(warnMsg, "should warn when plugins directory is missing");
  });

  it("warns when manifest fails validation", async () => {
    const tmpDir = await freshTmpDir();
    const invalidManifest = `
name: bad-plugin
version: 1.0.0
`;
    await createPluginDir(tmpDir, "bad-plugin", invalidManifest);

    const logger = mockLogger();
    const registry = new PluginRegistry(tmpDir, logger);
    await registry.load();

    const warnMsg = logger.calls.find(
      (c) => c.level === "warn" && c.message.includes("bad-plugin"),
    );
    assert.ok(warnMsg, "should warn with directory name when manifest is invalid");
  });

  it("warns when manifest YAML fails to parse", async () => {
    const tmpDir = await freshTmpDir();
    const brokenYaml = "name: [unterminated";
    await createPluginDir(tmpDir, "broken", brokenYaml);

    const logger = mockLogger();
    const registry = new PluginRegistry(tmpDir, logger);
    await registry.load();

    const warnMsg = logger.calls.find(
      (c) => c.level === "warn" && c.message.includes("broken"),
    );
    assert.ok(warnMsg, "should warn when YAML fails to parse");
  });

  it("works without a logger (backwards compatible)", async () => {
    const tmpDir = await freshTmpDir();
    await createPluginDir(tmpDir, "test-model", VALID_MANIFEST);

    // No logger argument — should not throw
    const registry = new PluginRegistry(tmpDir);
    await registry.load();

    assert.equal(registry.list().length, 1);
  });
});
