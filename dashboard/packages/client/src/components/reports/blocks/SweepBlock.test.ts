import { describe, it, expect } from "vitest";
import {
  extractSweepPanelData,
  extractInterpretabilitySummary,
} from "./SweepBlock.js";
import type { SweepAnalysis } from "@tidal/shared";

// ---------------------------------------------------------------------------
// Fixture: minimal sweep analysis (single modulation gate)
// ---------------------------------------------------------------------------

const fixtureSweep: SweepAnalysis = {
  configComparisons: {
    "0.0": {
      signalStats: {
        modulation: { mean: 0.0, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
      },
      textProperties: { wordCount: 10, uniqueTokenRatio: 0.7, charCount: 50 },
    },
    "0.5": {
      signalStats: {
        modulation: { mean: 0.5, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
      },
      textProperties: { wordCount: 15, uniqueTokenRatio: 0.75, charCount: 75 },
    },
    "1.0": {
      signalStats: {
        modulation: { mean: 1.0, std: 0, min: 0, max: 0, q25: 0, q50: 0, q75: 0 },
      },
      textProperties: { wordCount: 22, uniqueTokenRatio: 0.78, charCount: 110 },
    },
  },
  interpretabilityMap: {
    modulation: {
      lowConfig: "0.0",
      highConfig: "1.0",
      effect: {
        wordCount: { low: 10, high: 22, delta: 12 },
        uniqueTokenRatio: { low: 0.7, high: 0.78, delta: 0.08 },
        charCount: { low: 50, high: 110, delta: 60 },
      },
    },
  },
};

// ---------------------------------------------------------------------------
// extractSweepPanelData
// ---------------------------------------------------------------------------

describe("extractSweepPanelData", () => {
  it("returns one entry for the single modulation signal", () => {
    const result = extractSweepPanelData(fixtureSweep);
    expect(result).toHaveLength(1);
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

  it("modulation delta matches fixture", () => {
    const result = extractSweepPanelData(fixtureSweep);
    const modulation = result.find((r) => r.signal === "modulation");
    const wc = modulation?.properties.find((p) => p.name === "wordCount");
    expect(wc?.delta).toBe(12);
    expect(wc?.low).toBe(10);
    expect(wc?.high).toBe(22);
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
  it("returns one string for the single modulation signal", () => {
    const result = extractInterpretabilitySummary(fixtureSweep.interpretabilityMap);
    expect(result).toHaveLength(1);
  });

  it("string contains signal name", () => {
    const result = extractInterpretabilitySummary(fixtureSweep.interpretabilityMap);
    expect(result[0]).toContain("modulation");
  });

  it("includes delta direction indicators", () => {
    const result = extractInterpretabilitySummary(fixtureSweep.interpretabilityMap);
    // Modulation has positive wordCount delta
    expect(result[0]).toContain("+12");
  });

  it("handles empty interpretability map", () => {
    const result = extractInterpretabilitySummary({});
    expect(result).toEqual([]);
  });
});
