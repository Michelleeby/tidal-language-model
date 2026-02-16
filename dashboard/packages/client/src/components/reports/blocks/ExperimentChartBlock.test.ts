import { describe, it, expect } from "vitest";
import {
  extractValues,
  extractRLValues,
  extractAblationValues,
  extractAllAblationMetrics,
  metricOptionsForMode,
} from "./ExperimentChartBlock.js";
import type { RLTrainingHistory, AblationResults } from "@tidal/shared";

describe("extractValues", () => {
  const points = [
    { "Losses/Total": 2, "Learning Rate": 0.001, "Iterations/Second": 42 },
    { "Losses/Total": 1.5, "Learning Rate": 0.0008, "Iterations/Second": 45 },
    { "Losses/Total": 1, "Learning Rate": 0.0005, "Iterations/Second": 50 },
  ];

  it("extracts direct metric keys", () => {
    const vals = extractValues(
      points as Array<Record<string, unknown>>,
      "Losses/Total",
    );
    expect(vals).toEqual([2, 1.5, 1]);
  });

  it("extracts Learning Rate directly", () => {
    const vals = extractValues(
      points as Array<Record<string, unknown>>,
      "Learning Rate",
    );
    expect(vals).toEqual([0.001, 0.0008, 0.0005]);
  });

  it("computes Perplexity as exp(Losses/Total)", () => {
    const vals = extractValues(
      points as Array<Record<string, unknown>>,
      "Perplexity",
    );
    expect(vals).toEqual([Math.exp(2), Math.exp(1.5), Math.exp(1)]);
  });

  it("extracts Iterations/Second for throughput", () => {
    const vals = extractValues(
      points as Array<Record<string, unknown>>,
      "Iterations/Second",
    );
    expect(vals).toEqual([42, 45, 50]);
  });

  it("filters out NaN values", () => {
    const badPoints = [
      { "Losses/Total": 2 },
      { "Losses/Total": undefined },
      { "Losses/Total": 1 },
    ];
    const vals = extractValues(
      badPoints as Array<Record<string, unknown>>,
      "Losses/Total",
    );
    expect(vals).toEqual([2, 1]);
  });

  it("handles Perplexity with missing loss gracefully", () => {
    const badPoints = [
      { "Losses/Total": 2 },
      {}, // missing Losses/Total
      { "Losses/Total": 1 },
    ];
    const vals = extractValues(
      badPoints as Array<Record<string, unknown>>,
      "Perplexity",
    );
    expect(vals).toEqual([Math.exp(2), Math.exp(1)]);
  });
});

// ---------------------------------------------------------------------------
// extractRLValues
// ---------------------------------------------------------------------------

describe("extractRLValues", () => {
  const history: RLTrainingHistory = {
    episode_rewards: [1.0, 2.5, 3.0],
    policy_loss: [0.5, 0.3, 0.1],
    value_loss: [1.2, 0.8, 0.4],
    entropy: [0.9, 0.7, 0.5],
    gate_creativity: [0.6, 0.7, 0.8],
    gate_focus: [0.5, 0.55, 0.6],
    gate_stability: [0.9, 0.85, 0.8],
    reward_perplexity: [0.3, 0.4, 0.5],
    reward_diversity: [0.2, 0.25, 0.3],
    reward_repetition: [-0.1, -0.05, 0.0],
    reward_coherence: [0.4, 0.45, 0.5],
    explained_variance: [0.1, 0.3, 0.5],
  };

  it("extracts episode_rewards", () => {
    expect(extractRLValues(history, "episode_rewards")).toEqual([1.0, 2.5, 3.0]);
  });

  it("extracts gate_creativity", () => {
    expect(extractRLValues(history, "gate_creativity")).toEqual([0.6, 0.7, 0.8]);
  });

  it("extracts gate_focus", () => {
    expect(extractRLValues(history, "gate_focus")).toEqual([0.5, 0.55, 0.6]);
  });

  it("extracts gate_stability", () => {
    expect(extractRLValues(history, "gate_stability")).toEqual([0.9, 0.85, 0.8]);
  });

  it("extracts reward_perplexity", () => {
    expect(extractRLValues(history, "reward_perplexity")).toEqual([0.3, 0.4, 0.5]);
  });

  it("extracts reward_diversity", () => {
    expect(extractRLValues(history, "reward_diversity")).toEqual([0.2, 0.25, 0.3]);
  });

  it("extracts reward_repetition", () => {
    expect(extractRLValues(history, "reward_repetition")).toEqual([-0.1, -0.05, 0.0]);
  });

  it("extracts reward_coherence", () => {
    expect(extractRLValues(history, "reward_coherence")).toEqual([0.4, 0.45, 0.5]);
  });

  it("extracts explained_variance", () => {
    expect(extractRLValues(history, "explained_variance")).toEqual([0.1, 0.3, 0.5]);
  });

  it("extracts policy_loss", () => {
    expect(extractRLValues(history, "policy_loss")).toEqual([0.5, 0.3, 0.1]);
  });

  it("extracts value_loss", () => {
    expect(extractRLValues(history, "value_loss")).toEqual([1.2, 0.8, 0.4]);
  });

  it("extracts entropy", () => {
    expect(extractRLValues(history, "entropy")).toEqual([0.9, 0.7, 0.5]);
  });

  it("returns empty array for null history", () => {
    expect(extractRLValues(null, "episode_rewards")).toEqual([]);
  });

  it("returns empty array for unknown key", () => {
    expect(extractRLValues(history, "unknown_key")).toEqual([]);
  });

  it("filters out NaN values", () => {
    const badHistory: RLTrainingHistory = {
      episode_rewards: [1.0, NaN, 3.0],
      policy_loss: [],
      value_loss: [],
      entropy: [],
    };
    expect(extractRLValues(badHistory, "episode_rewards")).toEqual([1.0, 3.0]);
  });
});

// ---------------------------------------------------------------------------
// extractAblationValues
// ---------------------------------------------------------------------------

describe("extractAblationValues", () => {
  const results: AblationResults = {
    baseline: { mean_reward: 1.0, std_reward: 0.1, mean_diversity: 0.5, mean_perplexity: 50 },
    creative: { mean_reward: 2.0, std_reward: 0.2, mean_diversity: 0.8, mean_perplexity: 60 },
    focused: { mean_reward: 1.5, std_reward: 0.15, mean_diversity: 0.6, mean_perplexity: 45 },
  };

  it("extracts mean_reward with error bars", () => {
    const out = extractAblationValues(results, "mean_reward");
    expect(out.labels).toEqual(["baseline", "creative", "focused"]);
    expect(out.values).toEqual([1.0, 2.0, 1.5]);
    expect(out.errors).toEqual([0.1, 0.2, 0.15]);
  });

  it("extracts mean_diversity without error bars", () => {
    const out = extractAblationValues(results, "mean_diversity");
    expect(out.labels).toEqual(["baseline", "creative", "focused"]);
    expect(out.values).toEqual([0.5, 0.8, 0.6]);
    expect(out.errors).toBeUndefined();
  });

  it("extracts mean_perplexity without error bars", () => {
    const out = extractAblationValues(results, "mean_perplexity");
    expect(out.labels).toEqual(["baseline", "creative", "focused"]);
    expect(out.values).toEqual([50, 60, 45]);
    expect(out.errors).toBeUndefined();
  });

  it("returns empty arrays for null results", () => {
    const out = extractAblationValues(null, "mean_reward");
    expect(out.labels).toEqual([]);
    expect(out.values).toEqual([]);
    expect(out.errors).toBeUndefined();
  });

  it("handles single policy", () => {
    const single: AblationResults = {
      only: { mean_reward: 3.0, std_reward: 0.5, mean_diversity: 0.9, mean_perplexity: 30 },
    };
    const out = extractAblationValues(single, "mean_reward");
    expect(out.labels).toEqual(["only"]);
    expect(out.values).toEqual([3.0]);
    expect(out.errors).toEqual([0.5]);
  });
});

// ---------------------------------------------------------------------------
// extractAllAblationMetrics
// ---------------------------------------------------------------------------

describe("extractAllAblationMetrics", () => {
  const results: AblationResults = {
    baseline: { mean_reward: 1.0, std_reward: 0.1, mean_diversity: 0.5, mean_perplexity: 50 },
    creative: { mean_reward: 2.0, std_reward: 0.2, mean_diversity: 0.8, mean_perplexity: 60 },
  };

  it("returns all 3 metric series grouped by policy", () => {
    const out = extractAllAblationMetrics(results);
    expect(out.policies).toEqual(["baseline", "creative"]);
    expect(out.series).toHaveLength(3);

    expect(out.series[0].key).toBe("mean_reward");
    expect(out.series[0].values).toEqual([1.0, 2.0]);

    expect(out.series[1].key).toBe("mean_diversity");
    expect(out.series[1].values).toEqual([0.5, 0.8]);

    expect(out.series[2].key).toBe("mean_perplexity");
    expect(out.series[2].values).toEqual([50, 60]);
  });

  it("includes label and color for each series", () => {
    const out = extractAllAblationMetrics(results);
    expect(out.series[0].label).toBe("reward");
    expect(out.series[1].label).toBe("diversity");
    expect(out.series[2].label).toBe("perplexity");
    // Each series has a distinct color
    const colors = out.series.map((s) => s.color);
    expect(new Set(colors).size).toBe(3);
  });

  it("returns empty for null results", () => {
    const out = extractAllAblationMetrics(null);
    expect(out.policies).toEqual([]);
    expect(out.series).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// metricOptionsForMode
// ---------------------------------------------------------------------------

describe("metricOptionsForMode", () => {
  it("returns LM metric options", () => {
    const opts = metricOptionsForMode("lm");
    expect(opts).toEqual([
      { value: "Losses/Total", label: "Loss" },
      { value: "Learning Rate", label: "Learning Rate" },
      { value: "Perplexity", label: "Perplexity" },
      { value: "Iterations/Second", label: "Throughput" },
    ]);
  });

  it("returns RL metric options", () => {
    const opts = metricOptionsForMode("rl");
    expect(opts).toEqual([
      { value: "episode_rewards", label: "Episode Rewards" },
      { value: "policy_loss", label: "Policy Loss" },
      { value: "value_loss", label: "Value Loss" },
      { value: "entropy", label: "Entropy" },
      { value: "gate_creativity", label: "Gate: Creativity" },
      { value: "gate_focus", label: "Gate: Focus" },
      { value: "gate_stability", label: "Gate: Stability" },
      { value: "reward_perplexity", label: "Reward: Perplexity" },
      { value: "reward_diversity", label: "Reward: Diversity" },
      { value: "reward_repetition", label: "Reward: Repetition" },
      { value: "reward_coherence", label: "Reward: Coherence" },
      { value: "explained_variance", label: "Explained Variance" },
    ]);
  });

  it("returns ablation metric options", () => {
    const opts = metricOptionsForMode("ablation");
    expect(opts).toEqual([
      { value: "mean_reward", label: "Mean Reward" },
      { value: "mean_diversity", label: "Mean Diversity" },
      { value: "mean_perplexity", label: "Mean Perplexity" },
    ]);
  });

  it("falls back to LM options for unknown mode", () => {
    const opts = metricOptionsForMode("unknown" as any);
    expect(opts).toEqual(metricOptionsForMode("lm"));
  });
});
