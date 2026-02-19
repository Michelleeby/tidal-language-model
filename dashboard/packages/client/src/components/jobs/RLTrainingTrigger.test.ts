import { describe, it, expect } from "vitest";
import { filterRLEligibleCheckpoints } from "./RLTrainingTrigger.js";
import type { CheckpointInfo } from "@tidal/shared";

const makeCheckpoint = (
  overrides: Partial<CheckpointInfo> & { phase: string },
): CheckpointInfo => ({
  filename: `checkpoint_${overrides.phase}.pth`,
  path: `experiments/abc123/${overrides.phase}.pth`,
  sizeBytes: 1024,
  modified: Date.now(),
  ...overrides,
});

describe("filterRLEligibleCheckpoints", () => {
  it("includes foundational phase checkpoints", () => {
    const input = [makeCheckpoint({ phase: "foundational", epoch: 1 })];
    const result = filterRLEligibleCheckpoints(input);
    expect(result).toHaveLength(1);
    expect(result[0].phase).toBe("foundational");
  });

  it("includes final phase checkpoints", () => {
    const input = [makeCheckpoint({ phase: "final" })];
    const result = filterRLEligibleCheckpoints(input);
    expect(result).toHaveLength(1);
    expect(result[0].phase).toBe("final");
  });

  it("excludes rl phase checkpoints", () => {
    const input = [makeCheckpoint({ phase: "rl" })];
    const result = filterRLEligibleCheckpoints(input);
    expect(result).toHaveLength(0);
  });

  it("excludes unknown phase checkpoints", () => {
    const input = [makeCheckpoint({ phase: "unknown" })];
    const result = filterRLEligibleCheckpoints(input);
    expect(result).toHaveLength(0);
  });

  it("returns empty array for empty input", () => {
    const result = filterRLEligibleCheckpoints([]);
    expect(result).toHaveLength(0);
  });

  it("filters mixed input to only eligible checkpoints", () => {
    const input = [
      makeCheckpoint({ phase: "foundational", epoch: 1 }),
      makeCheckpoint({ phase: "final" }),
      makeCheckpoint({ phase: "rl" }),
      makeCheckpoint({ phase: "unknown" }),
      makeCheckpoint({ phase: "foundational", epoch: 2 }),
    ];
    const result = filterRLEligibleCheckpoints(input);
    expect(result).toHaveLength(3);
    expect(result.map((cp) => cp.phase)).toEqual([
      "foundational",
      "final",
      "foundational",
    ]);
  });
});
