import { describe, it, expect } from "vitest";
import {
  extractHeatmapData,
  extractCorrelationMatrix,
  extractVarianceSummary,
} from "./CrossPromptBlock.js";
import type { BatchAnalysis } from "@tidal/shared";

// ---------------------------------------------------------------------------
// Fixture: minimal batch analysis with 2 prompts
// ---------------------------------------------------------------------------

const makeSummary = (cMean: number, fMean: number, sMean: number) => ({
  signalStats: {
    creativity: { mean: cMean, std: 0.1, min: 0, max: 1, q25: 0.3, q50: 0.5, q75: 0.7 },
    focus: { mean: fMean, std: 0.1, min: 0, max: 1, q25: 0.3, q50: 0.5, q75: 0.7 },
    stability: { mean: sMean, std: 0.1, min: 0, max: 1, q25: 0.3, q50: 0.5, q75: 0.7 },
  },
  signalEvolution: {},
  crossSignalCorrelations: {
    creativity_focus: 0.5,
    creativity_stability: -0.3,
    focus_stability: 0.1,
  },
  phases: [],
  tokenSignalAlignment: {},
});

const fixtureBatch: BatchAnalysis = {
  perPromptSummaries: {
    "Once upon a time": makeSummary(0.8, 0.3, 0.5),
    "The cat sat": makeSummary(0.2, 0.7, 0.9),
  } as unknown as Record<string, unknown>,
  crossPromptVariance: {
    creativity: { betweenPromptVar: 0.09, withinPromptVar: 0.01 },
    focus: { betweenPromptVar: 0.04, withinPromptVar: 0.01 },
    stability: { betweenPromptVar: 0.04, withinPromptVar: 0.01 },
  },
  strategyCharacterization: {
    creativity: { globalMean: 0.5, globalStd: 0.3 },
    focus: { globalMean: 0.5, globalStd: 0.2 },
    stability: { globalMean: 0.7, globalStd: 0.2 },
  },
};

// ---------------------------------------------------------------------------
// extractHeatmapData
// ---------------------------------------------------------------------------

describe("extractHeatmapData", () => {
  it("returns prompts, signals, and values arrays", () => {
    const result = extractHeatmapData(fixtureBatch);
    expect(result.prompts).toHaveLength(2);
    expect(result.signals).toEqual(["creativity", "focus", "stability"]);
    expect(result.values).toHaveLength(2); // rows = prompts
    expect(result.values[0]).toHaveLength(3); // cols = signals
  });

  it("values match signal means from perPromptSummaries", () => {
    const result = extractHeatmapData(fixtureBatch);
    // "Once upon a time" → creativity=0.8, focus=0.3, stability=0.5
    const idx = result.prompts.indexOf("Once upon a time");
    expect(result.values[idx]).toEqual([0.8, 0.3, 0.5]);
  });

  it("handles empty perPromptSummaries", () => {
    const empty: BatchAnalysis = {
      perPromptSummaries: {} as Record<string, unknown>,
      crossPromptVariance: {},
      strategyCharacterization: {},
    };
    const result = extractHeatmapData(empty);
    expect(result.prompts).toEqual([]);
    expect(result.values).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// extractCorrelationMatrix
// ---------------------------------------------------------------------------

describe("extractCorrelationMatrix", () => {
  it("returns 3x3 matrix with signal labels", () => {
    const result = extractCorrelationMatrix(
      fixtureBatch.perPromptSummaries as Record<string, any>,
    );
    expect(result.labels).toEqual(["creativity", "focus", "stability"]);
    expect(result.matrix).toHaveLength(3);
    for (const row of result.matrix) {
      expect(row).toHaveLength(3);
    }
  });

  it("diagonal is 1.0", () => {
    const result = extractCorrelationMatrix(
      fixtureBatch.perPromptSummaries as Record<string, any>,
    );
    for (let i = 0; i < 3; i++) {
      expect(result.matrix[i][i]).toBe(1.0);
    }
  });

  it("is symmetric", () => {
    const result = extractCorrelationMatrix(
      fixtureBatch.perPromptSummaries as Record<string, any>,
    );
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(result.matrix[i][j]).toBe(result.matrix[j][i]);
      }
    }
  });

  it("handles empty summaries", () => {
    const result = extractCorrelationMatrix({});
    expect(result.labels).toEqual(["creativity", "focus", "stability"]);
    expect(result.matrix).toHaveLength(3);
    // All zeros when there are no prompts
    for (const row of result.matrix) {
      for (const v of row) {
        expect(v).toBe(v === 1.0 ? 1.0 : 0.0);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// extractVarianceSummary
// ---------------------------------------------------------------------------

describe("extractVarianceSummary", () => {
  it("returns one entry per signal", () => {
    const result = extractVarianceSummary(fixtureBatch.crossPromptVariance);
    expect(result).toHaveLength(3);
  });

  it("each entry has signal, between, within keys", () => {
    const result = extractVarianceSummary(fixtureBatch.crossPromptVariance);
    for (const entry of result) {
      expect(entry).toHaveProperty("signal");
      expect(entry).toHaveProperty("between");
      expect(entry).toHaveProperty("within");
    }
  });

  it("values match fixture", () => {
    const result = extractVarianceSummary(fixtureBatch.crossPromptVariance);
    const creativity = result.find((r) => r.signal === "creativity");
    expect(creativity?.between).toBe(0.09);
    expect(creativity?.within).toBe(0.01);
  });

  it("handles empty variance", () => {
    const result = extractVarianceSummary({});
    expect(result).toEqual([]);
  });
});
