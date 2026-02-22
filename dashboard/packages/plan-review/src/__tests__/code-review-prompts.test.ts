import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  getCodeReviewDimensionPrompt,
  getAllCodeReviewDimensions,
} from "../prompts/code-review-prompts.js";

// ---------------------------------------------------------------------------
// getCodeReviewDimensionPrompt
// ---------------------------------------------------------------------------

describe("getCodeReviewDimensionPrompt", () => {
  it("returns a non-empty prompt for each dimension", () => {
    for (const dim of ["bugs", "hypothesis_alignment", "adr_compliance"] as const) {
      const prompt = getCodeReviewDimensionPrompt(dim);
      assert.ok(prompt.length > 0, `Expected non-empty prompt for ${dim}`);
    }
  });

  it("each prompt contains the JSON schema instruction", () => {
    for (const dim of ["bugs", "hypothesis_alignment", "adr_compliance"] as const) {
      const prompt = getCodeReviewDimensionPrompt(dim);
      assert.ok(
        prompt.includes('"feedback"'),
        `Expected JSON schema "feedback" field in prompt for ${dim}`,
      );
      assert.ok(
        prompt.includes('"severity"'),
        `Expected JSON schema "severity" field in prompt for ${dim}`,
      );
    }
  });
});

// ---------------------------------------------------------------------------
// getAllCodeReviewDimensions
// ---------------------------------------------------------------------------

describe("getAllCodeReviewDimensions", () => {
  it("returns exactly 3 dimensions", () => {
    const dims = getAllCodeReviewDimensions();
    assert.equal(dims.length, 3);
  });

  it("includes all 3 expected dimensions", () => {
    const dims = getAllCodeReviewDimensions();
    assert.ok(dims.includes("bugs"));
    assert.ok(dims.includes("hypothesis_alignment"));
    assert.ok(dims.includes("adr_compliance"));
  });
});
