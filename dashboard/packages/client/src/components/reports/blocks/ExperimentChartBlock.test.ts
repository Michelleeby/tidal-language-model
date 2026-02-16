import { describe, it, expect } from "vitest";
import { extractValues } from "./ExperimentChartBlock.js";

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
