import { describe, it, expect } from "vitest";
import {
  extractHeatmapData,
  extractVarianceSummary,
} from "./CrossPromptBlock.js";
import type { BatchAnalysis } from "@tidal/shared";

// ---------------------------------------------------------------------------
// Fixture: minimal batch analysis with 2 prompts (single modulation gate)
// ---------------------------------------------------------------------------

const makeSummary = (mMean: number) => ({
  signalStats: {
    modulation: { mean: mMean, std: 0.1, min: 0, max: 1, q25: 0.3, q50: 0.5, q75: 0.7 },
  },
  signalEvolution: {},
  crossSignalCorrelations: {},
  phases: [],
  tokenSignalAlignment: {},
});

const fixtureBatch: BatchAnalysis = {
  perPromptSummaries: {
    "Once upon a time": makeSummary(0.6),
    "The cat sat": makeSummary(0.4),
  } as unknown as Record<string, unknown>,
  crossPromptVariance: {
    modulation: { betweenPromptVar: 0.01, withinPromptVar: 0.005 },
  },
  strategyCharacterization: {
    modulation: { globalMean: 0.5, globalStd: 0.1 },
  },
};

// ---------------------------------------------------------------------------
// extractHeatmapData
// ---------------------------------------------------------------------------

describe("extractHeatmapData", () => {
  it("returns prompts, signals, and values arrays", () => {
    const result = extractHeatmapData(fixtureBatch);
    expect(result.prompts).toHaveLength(2);
    expect(result.signals).toEqual(["modulation"]);
    expect(result.values).toHaveLength(2); // rows = prompts
    expect(result.values[0]).toHaveLength(1); // cols = 1 signal
  });

  it("values match signal means from perPromptSummaries", () => {
    const result = extractHeatmapData(fixtureBatch);
    const idx = result.prompts.indexOf("Once upon a time");
    expect(result.values[idx]).toEqual([0.6]);
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
// extractVarianceSummary
// ---------------------------------------------------------------------------

describe("extractVarianceSummary", () => {
  it("returns one entry for the single modulation signal", () => {
    const result = extractVarianceSummary(fixtureBatch.crossPromptVariance);
    expect(result).toHaveLength(1);
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
    const modulation = result.find((r) => r.signal === "modulation");
    expect(modulation?.between).toBe(0.01);
    expect(modulation?.within).toBe(0.005);
  });

  it("handles empty variance", () => {
    const result = extractVarianceSummary({});
    expect(result).toEqual([]);
  });
});
