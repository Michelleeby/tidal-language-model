import { describe, it, expect } from "vitest";
import {
  extractMiniChartData,
  extractMiniRLChartData,
  extractMiniAblationChartData,
  extractHeatmapRenderData,
  extractSweepRenderData,
  extractVarianceSummary,
  extractInterpretabilitySummary,
  renderHeatmap,
  renderSweepPanel,
  SIGNAL_COLORS,
  HEATMAP_DIMENSIONS,
  SWEEP_DIMENSIONS,
} from "./chartRenderers.js";
import type { BatchAnalysis, SweepAnalysis, RLTrainingHistory, AblationResults } from "@tidal/shared";

// ---------------------------------------------------------------------------
// extractMiniChartData
// ---------------------------------------------------------------------------

describe("extractMiniChartData", () => {
  it("extracts numeric values for a metric key", () => {
    const points = [
      { step: 0, "Losses/Total": 2.5 },
      { step: 1, "Losses/Total": 2.0 },
      { step: 2, "Losses/Total": 1.5 },
    ];
    const result = extractMiniChartData(points, "Losses/Total");
    expect(result).toEqual([2.5, 2.0, 1.5]);
  });

  it("computes Perplexity as exp(loss)", () => {
    const points = [
      { step: 0, "Losses/Total": 1.0 },
      { step: 1, "Losses/Total": 2.0 },
    ];
    const result = extractMiniChartData(points, "Perplexity");
    expect(result[0]).toBeCloseTo(Math.exp(1.0));
    expect(result[1]).toBeCloseTo(Math.exp(2.0));
  });

  it("filters NaN values", () => {
    const points = [
      { step: 0, "Losses/Total": 2.5 },
      { step: 1 }, // missing metric
      { step: 2, "Losses/Total": 1.5 },
    ];
    const result = extractMiniChartData(points, "Losses/Total");
    expect(result).toEqual([2.5, 1.5]);
  });
});

// ---------------------------------------------------------------------------
// extractMiniRLChartData
// ---------------------------------------------------------------------------

describe("extractMiniRLChartData", () => {
  it("extracts RL metric array", () => {
    const history = {
      episode_rewards: [1.0, 2.0, 3.0],
      policy_loss: [],
      value_loss: [],
      entropy: [],
    } as unknown as RLTrainingHistory;

    const result = extractMiniRLChartData(history, "episode_rewards");
    expect(result).toEqual([1.0, 2.0, 3.0]);
  });

  it("returns empty array for null history", () => {
    const result = extractMiniRLChartData(null, "episode_rewards");
    expect(result).toEqual([]);
  });

  it("returns empty array for unknown key", () => {
    const history = {
      episode_rewards: [1.0],
    } as unknown as RLTrainingHistory;
    const result = extractMiniRLChartData(history, "unknown_key");
    expect(result).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// extractMiniAblationChartData
// ---------------------------------------------------------------------------

describe("extractMiniAblationChartData", () => {
  it("extracts policies and series from ablation results", () => {
    const results: AblationResults = {
      none: { mean_reward: 1.0, std_reward: 0.1, mean_diversity: 0.8, mean_perplexity: 50 },
      fixed: { mean_reward: 1.5, std_reward: 0.2, mean_diversity: 0.9, mean_perplexity: 45 },
    } as unknown as AblationResults;

    const data = extractMiniAblationChartData(results);
    expect(data.policies).toEqual(["none", "fixed"]);
    expect(data.series).toHaveLength(3); // reward, diversity, perplexity
    expect(data.series[0].key).toBe("mean_reward");
    expect(data.series[0].values).toEqual([1.0, 1.5]);
  });

  it("returns empty data for null results", () => {
    const data = extractMiniAblationChartData(null);
    expect(data.policies).toEqual([]);
    expect(data.series).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// extractHeatmapRenderData
// ---------------------------------------------------------------------------

describe("extractHeatmapRenderData", () => {
  it("extracts heatmap data from batch analysis", () => {
    const batch: BatchAnalysis = {
      perPromptSummaries: {
        "prompt A": {
          signalStats: { modulation: { mean: 0.6, std: 0.1, min: 0, max: 1, q25: 0.3, q50: 0.5, q75: 0.7 } },
        },
      } as unknown as Record<string, unknown>,
      crossPromptVariance: {},
      strategyCharacterization: {},
    };

    const result = extractHeatmapRenderData(batch);
    expect(result.prompts).toEqual(["prompt A"]);
    expect(result.signals).toEqual(["modulation"]);
    expect(result.values[0][0]).toBe(0.6);
  });
});

// ---------------------------------------------------------------------------
// extractSweepRenderData
// ---------------------------------------------------------------------------

describe("extractSweepRenderData", () => {
  it("extracts sweep panel data from sweep analysis", () => {
    const sweep: SweepAnalysis = {
      configComparisons: {},
      interpretabilityMap: {
        modulation: {
          lowConfig: "low",
          highConfig: "high",
          effect: {
            wordCount: { low: 5, high: 15, delta: 10 },
            uniqueTokenRatio: { low: 0.3, high: 0.7, delta: 0.4 },
            charCount: { low: 20, high: 80, delta: 60 },
          },
        },
      },
    };

    const result = extractSweepRenderData(sweep);
    expect(result).toHaveLength(1);
    expect(result[0].signal).toBe("modulation");
    expect(result[0].properties).toHaveLength(3);
    expect(result[0].properties[0].name).toBe("wordCount");
  });

  it("returns empty array for empty interpretability map", () => {
    const sweep: SweepAnalysis = {
      configComparisons: {},
      interpretabilityMap: {},
    };
    const result = extractSweepRenderData(sweep);
    expect(result).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// extractVarianceSummary
// ---------------------------------------------------------------------------

describe("extractVarianceSummary", () => {
  it("extracts variance entries per signal", () => {
    const variance = {
      modulation: { betweenPromptVar: 0.01, withinPromptVar: 0.005 },
    };
    const result = extractVarianceSummary(variance);
    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({ signal: "modulation", between: 0.01, within: 0.005 });
  });

  it("returns empty for empty object", () => {
    expect(extractVarianceSummary({})).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// extractInterpretabilitySummary
// ---------------------------------------------------------------------------

describe("extractInterpretabilitySummary", () => {
  it("returns human-readable summary strings", () => {
    const imap = {
      modulation: {
        lowConfig: "0.0",
        highConfig: "1.0",
        effect: {
          wordCount: { low: 10, high: 22, delta: 12 },
          uniqueTokenRatio: { low: 0.7, high: 0.78, delta: 0.08 },
          charCount: { low: 50, high: 110, delta: 60 },
        },
      },
    };
    const result = extractInterpretabilitySummary(imap);
    expect(result).toHaveLength(1);
    expect(result[0]).toContain("modulation");
    expect(result[0]).toContain("+12");
  });

  it("returns empty for empty or missing map", () => {
    expect(extractInterpretabilitySummary({})).toEqual([]);
    expect(extractInterpretabilitySummary(undefined as any)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

describe("shared constants", () => {
  it("exports SIGNAL_COLORS", () => {
    expect(SIGNAL_COLORS).toContain("#a78bfa");
  });

  it("exports dimension constants", () => {
    expect(HEATMAP_DIMENSIONS.width).toBe(600);
    expect(HEATMAP_DIMENSIONS.height).toBe(300);
    expect(SWEEP_DIMENSIONS.width).toBe(600);
    expect(SWEEP_DIMENSIONS.rowHeight).toBe(80);
    expect(SWEEP_DIMENSIONS.headerHeight).toBe(24);
  });
});

// ---------------------------------------------------------------------------
// Canvas rendering functions — edge case safety
// ---------------------------------------------------------------------------

describe("renderHeatmap", () => {
  it("is a no-op when canvas is null", () => {
    // Should not throw
    renderHeatmap(null, { prompts: ["a"], signals: ["s"], values: [[0.5]] });
  });

  it("is a no-op when data has no prompts", () => {
    renderHeatmap(null, { prompts: [], signals: [], values: [] });
  });
});

describe("renderSweepPanel", () => {
  it("is a no-op when canvas is null", () => {
    renderSweepPanel(null, [{ signal: "mod", properties: [{ name: "wc", low: 1, high: 2, delta: 1 }] }]);
  });

  it("is a no-op when data is empty", () => {
    renderSweepPanel(null, []);
  });
});
