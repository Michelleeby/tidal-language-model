import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { aggregateFeedback, jaccardSimilarity } from "../feedback/feedback-aggregator.js";
import type { FeedbackItem, AggregatedFeedbackItem } from "../types.js";

// ---------------------------------------------------------------------------
// jaccardSimilarity
// ---------------------------------------------------------------------------

describe("jaccardSimilarity", () => {
  it("returns 1.0 for identical strings", () => {
    const score = jaccardSimilarity("hello world", "hello world");
    assert.equal(score, 1.0);
  });

  it("returns 0.0 for completely different strings", () => {
    const score = jaccardSimilarity("apple banana", "cherry durian");
    assert.equal(score, 0.0);
  });

  it("returns value between 0 and 1 for partial overlap", () => {
    const score = jaccardSimilarity("the quick brown fox", "the slow brown dog");
    assert.ok(score > 0);
    assert.ok(score < 1);
  });

  it("is case-insensitive", () => {
    const score = jaccardSimilarity("Hello World", "hello world");
    assert.equal(score, 1.0);
  });

  it("ignores punctuation", () => {
    const score = jaccardSimilarity("hello, world!", "hello world");
    assert.equal(score, 1.0);
  });
});

// ---------------------------------------------------------------------------
// aggregateFeedback
// ---------------------------------------------------------------------------

describe("aggregateFeedback", () => {
  const item1: FeedbackItem = {
    severity: "warning",
    description: "Missing error handling for network failures in the API client",
    affected_files: ["src/api-client.ts"],
    reasoning: "Network calls can fail",
    dimension: "completeness",
    source: "openai",
  };

  const item2: FeedbackItem = {
    severity: "warning",
    description: "No error handling for network failures in the API client module",
    affected_files: ["src/api-client.ts"],
    reasoning: "HTTP requests need error handling",
    dimension: "completeness",
    source: "google",
  };

  const item3: FeedbackItem = {
    severity: "critical",
    description: "Checkpoint format change will break existing saved models",
    affected_files: ["src/checkpoint.py"],
    reasoning: "Format incompatibility",
    dimension: "regression_risk",
    source: "anthropic",
  };

  it("deduplicates similar items using Jaccard threshold", () => {
    const result = aggregateFeedback([item1, item2, item3]);
    // item1 and item2 are similar — should be merged
    assert.equal(result.length, 2);
  });

  it("keeps the longest description when merging", () => {
    const result = aggregateFeedback([item1, item2]);
    assert.equal(result.length, 1);
    // item2 has longer description
    assert.ok(result[0].description.length >= item1.description.length);
  });

  it("unions affected_files from merged items", () => {
    const itemA: FeedbackItem = {
      ...item1,
      affected_files: ["file-a.ts"],
    };
    const itemB: FeedbackItem = {
      ...item2,
      affected_files: ["file-b.ts"],
    };
    const result = aggregateFeedback([itemA, itemB]);
    assert.equal(result.length, 1);
    assert.ok(result[0].affected_files.includes("file-a.ts"));
    assert.ok(result[0].affected_files.includes("file-b.ts"));
  });

  it("records corroborated_by sources", () => {
    const result = aggregateFeedback([item1, item2]);
    assert.equal(result.length, 1);
    assert.ok(result[0].corroborated_by.includes("openai"));
    assert.ok(result[0].corroborated_by.includes("google"));
  });

  it("promotes severity with corroboration", () => {
    const result = aggregateFeedback([item1, item2]);
    assert.equal(result.length, 1);
    // warning + 1 extra source → critical
    assert.equal(result[0].severity, "critical");
  });

  it("does not merge items from different dimensions even if similar text", () => {
    const itemDiffDim: FeedbackItem = {
      ...item1,
      dimension: "blind_spots",
      source: "google",
    };
    const result = aggregateFeedback([item1, itemDiffDim]);
    assert.equal(result.length, 2);
  });

  it("calculates ranking score correctly", () => {
    const result = aggregateFeedback([item1, item2, item3]);
    // item3 is critical (3.0) + file specificity (0.5) = 3.5
    // merged item1+2 is critical after promotion (3.0) + corroboration (1.0) + file (0.5) = 4.5
    // So merged item should rank first
    assert.ok(result[0].score >= result[1].score);
  });

  it("sorts by score descending", () => {
    const items: FeedbackItem[] = [
      { severity: "suggestion", description: "Minor style issue", affected_files: [], reasoning: "Style", dimension: "completeness", source: "openai" },
      { severity: "critical", description: "Security vulnerability in auth", affected_files: ["auth.ts"], reasoning: "Security", dimension: "blind_spots", source: "google" },
      { severity: "warning", description: "Missing test for edge case", affected_files: ["test.ts"], reasoning: "Coverage", dimension: "test_coverage", source: "anthropic" },
    ];
    const result = aggregateFeedback(items);
    for (let i = 1; i < result.length; i++) {
      assert.ok(result[i - 1].score >= result[i].score);
    }
  });

  it("handles empty input", () => {
    const result = aggregateFeedback([]);
    assert.equal(result.length, 0);
  });

  it("caps severity promotion at critical", () => {
    // 3 sources all saying the same critical thing
    const items: FeedbackItem[] = [
      { severity: "critical", description: "Critical breaking change in API", affected_files: ["api.ts"], reasoning: "Break", dimension: "regression_risk", source: "openai" },
      { severity: "critical", description: "Critical breaking change in API endpoint", affected_files: ["api.ts"], reasoning: "Break", dimension: "regression_risk", source: "google" },
      { severity: "critical", description: "Critical breaking change in the API contract", affected_files: ["api.ts"], reasoning: "Break", dimension: "regression_risk", source: "anthropic" },
    ];
    const result = aggregateFeedback(items);
    assert.equal(result.length, 1);
    assert.equal(result[0].severity, "critical"); // stays critical, no higher
  });
});
