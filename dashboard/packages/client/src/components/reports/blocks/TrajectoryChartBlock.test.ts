import { describe, it, expect } from "vitest";
import { gatingModeOptions } from "./TrajectoryChartBlock.js";

describe("gatingModeOptions", () => {
  it("returns all 3 gating modes", () => {
    const opts = gatingModeOptions();
    expect(opts).toHaveLength(3);
  });

  it("includes fixed, random, learned", () => {
    const opts = gatingModeOptions();
    const ids = opts.map((o) => o.value);
    expect(ids).toContain("fixed");
    expect(ids).toContain("random");
    expect(ids).toContain("learned");
  });

  it("each option has a label", () => {
    const opts = gatingModeOptions();
    for (const opt of opts) {
      expect(opt.label).toBeTruthy();
    }
  });
});
