import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { compactContext, estimateTokens } from "../context/context-compactor.js";

// ---------------------------------------------------------------------------
// estimateTokens
// ---------------------------------------------------------------------------

describe("estimateTokens", () => {
  it("estimates ~1 token per 4 characters", () => {
    assert.equal(estimateTokens("abcd"), 1);
    assert.equal(estimateTokens("abcde"), 2);
    assert.equal(estimateTokens(""), 0);
  });

  it("handles long text", () => {
    const text = "a".repeat(1000);
    assert.equal(estimateTokens(text), 250);
  });
});

// ---------------------------------------------------------------------------
// compactContext
// ---------------------------------------------------------------------------

describe("compactContext", () => {
  it("returns all inputs when within budget", () => {
    const result = compactContext({
      plan: "Short plan",
      adrSummaries: "ADR 1: Short summary",
      codeContext: "function foo() {}",
      tokenBudget: 5000,
    });
    assert.ok(result.includes("Short plan"));
    assert.ok(result.includes("ADR 1: Short summary"));
    assert.ok(result.includes("function foo() {}"));
    assert.equal(result.truncated, false);
  });

  it("strips extra blank lines", () => {
    const result = compactContext({
      plan: "Line 1\n\n\n\n\nLine 2",
      adrSummaries: "",
      codeContext: "",
      tokenBudget: 5000,
    });
    assert.ok(!result.includes("\n\n\n"));
  });

  it("removes horizontal rules", () => {
    const result = compactContext({
      plan: "Before\n\n---\n\nAfter",
      adrSummaries: "",
      codeContext: "",
      tokenBudget: 5000,
    });
    assert.ok(!result.includes("---"));
    assert.ok(result.includes("Before"));
    assert.ok(result.includes("After"));
  });

  it("collapses code block bodies to signatures when over budget", () => {
    const longCode = "```python\ndef train_model(config):\n    x = 1\n    y = 2\n    z = 3\n    return x + y + z\n```";
    const result = compactContext({
      plan: "Plan text here",
      adrSummaries: "",
      codeContext: longCode,
      tokenBudget: 30, // tight budget forces compaction
    });
    // Should still contain the function signature
    assert.ok(result.includes("train_model"));
  });

  it("drops ADR alternatives sections when over budget", () => {
    const adrWithAlternatives = `ADR 1: Title.
## Alternatives Considered
### Option A
Long description of option A that we rejected.
### Option B
Long description of option B that we rejected.
## References
- file.py`;
    const result = compactContext({
      plan: "Short plan",
      adrSummaries: adrWithAlternatives,
      codeContext: "",
      tokenBudget: 30,
    });
    assert.ok(!result.includes("Option A"));
    assert.ok(!result.includes("Option B"));
  });

  it("preserves plan content as highest priority", () => {
    const longPlan = "Important plan content. ".repeat(100);
    const result = compactContext({
      plan: longPlan,
      adrSummaries: "ADR summary",
      codeContext: "code context",
      tokenBudget: 100,
    });
    assert.ok(result.includes("Important plan content"));
  });

  it("adds truncation marker when plan is truncated", () => {
    const longPlan = "Word ".repeat(5000);
    const result = compactContext({
      plan: longPlan,
      adrSummaries: "",
      codeContext: "",
      tokenBudget: 50,
    });
    assert.equal(result.truncated, true);
    assert.ok(result.includes("[truncated]"));
  });
});
