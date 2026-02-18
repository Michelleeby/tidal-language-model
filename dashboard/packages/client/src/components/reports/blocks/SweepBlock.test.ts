import { describe, it, expect } from "vitest";
import {
  extractSweepPanelData,
  extractInterpretabilitySummary,
} from "./SweepBlock.js";
import type { SweepAnalysis } from "@tidal/shared";

// ---------------------------------------------------------------------------
// Fixture: minimal sweep analysis
// ---------------------------------------------------------------------------

const fixtureSweep: SweepAnalysis = {
  configComparisons: {
    "0.0_0.5_0.5": {
      signalStats: {
        creativity: { mean: 0.0, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
        focus: { mean: 0.5, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
        stability: { mean: 0.5, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
      },
      textProperties: { wordCount: 10, uniqueTokenRatio: 0.7, charCount: 50 },
    },
    "1.0_0.5_0.5": {
      signalStats: {
        creativity: { mean: 1.0, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
        focus: { mean: 0.5, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
        stability: { mean: 0.5, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
      },
      textProperties: { wordCount: 22, uniqueTokenRatio: 0.78, charCount: 110 },
    },
    "0.5_0.0_0.5": {
      signalStats: {
        creativity: { mean: 0.5, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
        focus: { mean: 0.0, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
        stability: { mean: 0.5, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
      },
      textProperties: { wordCount: 15, uniqueTokenRatio: 0.6, charCount: 75 },
    },
    "0.5_1.0_0.5": {
      signalStats: {
        creativity: { mean: 0.5, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
        focus: { mean: 1.0, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
        stability: { mean: 0.5, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
      },
      textProperties: { wordCount: 18, uniqueTokenRatio: 0.65, charCount: 90 },
    },
    "0.5_0.5_0.0": {
      signalStats: {
        creativity: { mean: 0.5, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
        focus: { mean: 0.5, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
        stability: { mean: 0.0, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
      },
      textProperties: { wordCount: 12, uniqueTokenRatio: 0.72, charCount: 60 },
    },
    "0.5_0.5_1.0": {
      signalStats: {
        creativity: { mean: 0.5, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
        focus: { mean: 0.5, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
        stability: { mean: 1.0, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
      },
      textProperties: { wordCount: 8, uniqueTokenRatio: 0.85, charCount: 40 },
    },
  },
  interpretabilityMap: {
    creativity: {
      lowConfig: "0.0_0.5_0.5",
      highConfig: "1.0_0.5_0.5",
      effect: {
        wordCount: { low: 10, high: 22, delta: 12 },
        uniqueTokenRatio: { low: 0.7, high: 0.78, delta: 0.08 },
        charCount: { low: 50, high: 110, delta: 60 },
      },
    },
    focus: {
      lowConfig: "0.5_0.0_0.5",
      highConfig: "0.5_1.0_0.5",
      effect: {
        wordCount: { low: 15, high: 18, delta: 3 },
        uniqueTokenRatio: { low: 0.6, high: 0.65, delta: 0.05 },
        charCount: { low: 75, high: 90, delta: 15 },
      },
    },
    stability: {
      lowConfig: "0.5_0.5_0.0",
      highConfig: "0.5_0.5_1.0",
      effect: {
        wordCount: { low: 12, high: 8, delta: -4 },
        uniqueTokenRatio: { low: 0.72, high: 0.85, delta: 0.13 },
        charCount: { low: 60, high: 40, delta: -20 },
      },
    },
  },
};

// ---------------------------------------------------------------------------
// extractSweepPanelData
// ---------------------------------------------------------------------------

describe("extractSweepPanelData", () => {
  it("returns one entry per signal", () => {
    const result = extractSweepPanelData(fixtureSweep);
    expect(result).toHaveLength(3);
  });

  it("each entry has signal, properties with low/high/delta", () => {
    const result = extractSweepPanelData(fixtureSweep);
    for (const entry of result) {
      expect(entry).toHaveProperty("signal");
      expect(entry).toHaveProperty("properties");
      for (const prop of entry.properties) {
        expect(prop).toHaveProperty("name");
        expect(prop).toHaveProperty("low");
        expect(prop).toHaveProperty("high");
        expect(prop).toHaveProperty("delta");
      }
    }
  });

  it("creativity delta matches fixture", () => {
    const result = extractSweepPanelData(fixtureSweep);
    const creativity = result.find((r) => r.signal === "creativity");
    const wc = creativity?.properties.find((p) => p.name === "wordCount");
    expect(wc?.delta).toBe(12);
    expect(wc?.low).toBe(10);
    expect(wc?.high).toBe(22);
  });

  it("stability has negative wordCount delta", () => {
    const result = extractSweepPanelData(fixtureSweep);
    const stability = result.find((r) => r.signal === "stability");
    const wc = stability?.properties.find((p) => p.name === "wordCount");
    expect(wc?.delta).toBe(-4);
  });

  it("handles empty sweep analysis", () => {
    const empty: SweepAnalysis = {
      configComparisons: {},
      interpretabilityMap: {},
    };
    const result = extractSweepPanelData(empty);
    expect(result).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// extractInterpretabilitySummary
// ---------------------------------------------------------------------------

describe("extractInterpretabilitySummary", () => {
  it("returns one string per signal", () => {
    const result = extractInterpretabilitySummary(fixtureSweep.interpretabilityMap);
    expect(result).toHaveLength(3);
  });

  it("each string contains signal name", () => {
    const result = extractInterpretabilitySummary(fixtureSweep.interpretabilityMap);
    expect(result[0]).toContain("creativity");
    expect(result[1]).toContain("focus");
    expect(result[2]).toContain("stability");
  });

  it("includes delta direction indicators", () => {
    const result = extractInterpretabilitySummary(fixtureSweep.interpretabilityMap);
    // Creativity has positive wordCount delta
    expect(result[0]).toContain("+12");
    // Stability has negative wordCount delta
    expect(result[2]).toContain("-4");
  });

  it("handles empty interpretability map", () => {
    const result = extractInterpretabilitySummary({});
    expect(result).toEqual([]);
  });
});
