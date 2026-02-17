import { describe, it, expect } from "vitest";
import { buildTrajectoryChartData } from "./GateTrajectoryChart.js";
import type { GenerationTrajectory } from "@tidal/shared";

const makeTraj = (len: number): GenerationTrajectory => ({
  gateSignals: Array.from({ length: len }, (_, i) => [
    0.1 * i,
    0.2 * i,
    0.3 * i,
  ]) as [number, number, number][],
  effects: Array.from({ length: len }, (_, i) => ({
    temperature: 0.5 + 0.1 * i,
    repetition_penalty: 1.0 + 0.1 * i,
    top_k: 10 + i,
    top_p: 0.8 + 0.01 * i,
  })),
  tokenIds: Array.from({ length: len }, (_, i) => 100 + i),
  tokenTexts: Array.from({ length: len }, (_, i) => `tok${i}`),
});

describe("buildTrajectoryChartData", () => {
  it("returns correct series count for signals mode", () => {
    const traj = makeTraj(5);
    const result = buildTrajectoryChartData(traj, "signals");
    // steps + creativity + focus + stability = 4 arrays
    expect(result).toHaveLength(4);
  });

  it("returns correct series count for effects mode", () => {
    const traj = makeTraj(5);
    const result = buildTrajectoryChartData(traj, "effects");
    // steps + temperature + rep_penalty + top_k + top_p = 5 arrays
    expect(result).toHaveLength(5);
  });

  it("extracts gate signal values correctly", () => {
    const traj = makeTraj(3);
    const result = buildTrajectoryChartData(traj, "signals");
    const steps = result[0];
    const creativity = result[1];
    const focus = result[2];
    const stability = result[3];

    expect(Array.from(steps)).toEqual([0, 1, 2]);
    expect(Array.from(creativity)).toEqual([0.0, 0.1, 0.2]);
    expect(Array.from(focus)).toEqual([0.0, 0.2, 0.4]);
    expect(Array.from(stability)).toEqual([0.0, 0.3, 0.6]);
  });

  it("returns empty arrays for null trajectory", () => {
    const result = buildTrajectoryChartData(null, "signals");
    expect(result).toHaveLength(4);
    for (const arr of result) {
      expect(arr).toHaveLength(0);
    }
  });
});
