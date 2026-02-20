import { describe, it, expect } from "vitest";
import { hasCompletedLMExperiment } from "./experimentFilters.js";
import type { ExperimentSummary } from "@tidal/shared";

function makeExperiment(
  overrides: Partial<ExperimentSummary> = {},
): ExperimentSummary {
  return {
    id: "exp-001",
    path: "/experiments/exp-001",
    created: Date.now(),
    hasRLMetrics: false,
    hasEvaluation: false,
    hasAblation: false,
    hasGpuInstance: false,
    status: null,
    checkpoints: [],
    experimentType: "unknown",
    sourceExperimentId: null,
    sourceCheckpoint: null,
    ...overrides,
  };
}

describe("hasCompletedLMExperiment", () => {
  it("returns true when a completed LM experiment with checkpoints exists", () => {
    const experiments = [
      makeExperiment({
        experimentType: "lm",
        status: { status: "completed", last_update: 0 },
        checkpoints: ["checkpoint_foundational_epoch_1.pth"],
      }),
    ];
    expect(hasCompletedLMExperiment(experiments)).toBe(true);
  });

  it("returns false when LM experiment is still training", () => {
    const experiments = [
      makeExperiment({
        experimentType: "lm",
        status: { status: "training", last_update: 0 },
        checkpoints: ["checkpoint_foundational_epoch_1.pth"],
      }),
    ];
    expect(hasCompletedLMExperiment(experiments)).toBe(false);
  });

  it("returns false when LM experiment is initialized", () => {
    const experiments = [
      makeExperiment({
        experimentType: "lm",
        status: { status: "initialized", last_update: 0 },
        checkpoints: [],
      }),
    ];
    expect(hasCompletedLMExperiment(experiments)).toBe(false);
  });

  it("returns false when completed experiment is RL type", () => {
    const experiments = [
      makeExperiment({
        experimentType: "rl",
        status: { status: "completed", last_update: 0 },
        checkpoints: ["rl_agent.pth"],
      }),
    ];
    expect(hasCompletedLMExperiment(experiments)).toBe(false);
  });

  it("returns false when no experiments exist", () => {
    expect(hasCompletedLMExperiment([])).toBe(false);
  });

  it("returns false when completed LM experiment has no checkpoints", () => {
    const experiments = [
      makeExperiment({
        experimentType: "lm",
        status: { status: "completed", last_update: 0 },
        checkpoints: [],
      }),
    ];
    expect(hasCompletedLMExperiment(experiments)).toBe(false);
  });

  it("returns true when at least one of several experiments qualifies", () => {
    const experiments = [
      makeExperiment({
        id: "exp-training",
        experimentType: "lm",
        status: { status: "training", last_update: 0 },
        checkpoints: [],
      }),
      makeExperiment({
        id: "exp-rl",
        experimentType: "rl",
        status: { status: "completed", last_update: 0 },
        checkpoints: ["rl.pth"],
      }),
      makeExperiment({
        id: "exp-done",
        experimentType: "lm",
        status: { status: "completed", last_update: 0 },
        checkpoints: ["checkpoint.pth"],
      }),
    ];
    expect(hasCompletedLMExperiment(experiments)).toBe(true);
  });

  it("returns false when status is null", () => {
    const experiments = [
      makeExperiment({
        experimentType: "lm",
        status: null,
        checkpoints: ["checkpoint.pth"],
      }),
    ];
    expect(hasCompletedLMExperiment(experiments)).toBe(false);
  });
});
