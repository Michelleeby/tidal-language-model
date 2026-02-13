import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { TrainingJob, JobType } from "@tidal/shared";
import {
  LMTrainingPolicy,
  RLTrainingPolicy,
  JobPolicyRegistry,
  type JobPolicy,
} from "../job-policy.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeJob(
  overrides: Partial<TrainingJob> & { type: JobType },
): TrainingJob {
  return {
    jobId: "test-job-" + Math.random().toString(36).slice(2, 8),
    status: "running",
    provider: "local",
    config: { type: overrides.type, configPath: "configs/base_config.yaml" },
    createdAt: Date.now(),
    updatedAt: Date.now(),
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// LMTrainingPolicy
// ---------------------------------------------------------------------------

describe("LMTrainingPolicy", () => {
  const policy = new LMTrainingPolicy();

  it("has type 'lm-training'", () => {
    assert.equal(policy.type, "lm-training");
  });

  it("returns null (no conflict) when no active jobs", () => {
    assert.equal(policy.checkConcurrency([]), null);
  });

  it("returns null when only RL jobs are active", () => {
    const jobs = [makeJob({ type: "rl-training" })];
    assert.equal(policy.checkConcurrency(jobs), null);
  });

  it("returns error string when an LM job is already active", () => {
    const jobs = [makeJob({ type: "lm-training" })];
    const result = policy.checkConcurrency(jobs);
    assert.ok(typeof result === "string");
    assert.ok(result.includes("already running"));
  });

  it("returns 'standard' GPU tier", () => {
    assert.equal(policy.gpuTier(), "standard");
  });
});

// ---------------------------------------------------------------------------
// RLTrainingPolicy
// ---------------------------------------------------------------------------

describe("RLTrainingPolicy", () => {
  const policy = new RLTrainingPolicy();

  it("has type 'rl-training'", () => {
    assert.equal(policy.type, "rl-training");
  });

  it("returns null (no conflict) when no active jobs", () => {
    assert.equal(policy.checkConcurrency([]), null);
  });

  it("returns null when only LM jobs are active", () => {
    const jobs = [makeJob({ type: "lm-training" })];
    assert.equal(policy.checkConcurrency(jobs), null);
  });

  it("returns error string when an RL job is already active", () => {
    const jobs = [makeJob({ type: "rl-training" })];
    const result = policy.checkConcurrency(jobs);
    assert.ok(typeof result === "string");
    assert.ok(result.includes("already running"));
  });

  it("returns 'standard' GPU tier", () => {
    assert.equal(policy.gpuTier(), "standard");
  });
});

// ---------------------------------------------------------------------------
// Coexistence: LM + RL can run simultaneously
// ---------------------------------------------------------------------------

describe("LM + RL coexistence", () => {
  it("LM policy allows start when RL job is active", () => {
    const lmPolicy = new LMTrainingPolicy();
    const activeJobs = [makeJob({ type: "rl-training" })];
    assert.equal(lmPolicy.checkConcurrency(activeJobs), null);
  });

  it("RL policy allows start when LM job is active", () => {
    const rlPolicy = new RLTrainingPolicy();
    const activeJobs = [makeJob({ type: "lm-training" })];
    assert.equal(rlPolicy.checkConcurrency(activeJobs), null);
  });

  it("both block same-type concurrent jobs", () => {
    const lm = new LMTrainingPolicy();
    const rl = new RLTrainingPolicy();
    const bothActive = [
      makeJob({ type: "lm-training" }),
      makeJob({ type: "rl-training" }),
    ];
    assert.ok(lm.checkConcurrency(bothActive) !== null);
    assert.ok(rl.checkConcurrency(bothActive) !== null);
  });
});

// ---------------------------------------------------------------------------
// JobPolicyRegistry
// ---------------------------------------------------------------------------

describe("JobPolicyRegistry", () => {
  it("returns registered policy by type", () => {
    const registry = new JobPolicyRegistry();
    const lm = registry.get("lm-training");
    assert.ok(lm);
    assert.equal(lm.type, "lm-training");
  });

  it("returns registered RL policy", () => {
    const registry = new JobPolicyRegistry();
    const rl = registry.get("rl-training");
    assert.ok(rl);
    assert.equal(rl.type, "rl-training");
  });

  it("returns undefined for unregistered type", () => {
    const registry = new JobPolicyRegistry();
    const result = registry.get("unknown" as JobType);
    assert.equal(result, undefined);
  });

  it("supports custom policy registration", () => {
    const registry = new JobPolicyRegistry();
    const custom: JobPolicy = {
      type: "lm-training" as JobType,
      checkConcurrency: () => "blocked",
      gpuTier: () => "standard",
    };
    registry.register(custom);
    assert.equal(registry.get("lm-training"), custom);
  });
});
