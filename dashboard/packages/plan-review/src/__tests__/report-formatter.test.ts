import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { formatReviewReport } from "../report-formatter.js";
import type { AggregatedFeedbackItem } from "../types.js";

// ---------------------------------------------------------------------------
// formatReviewReport
// ---------------------------------------------------------------------------

describe("formatReviewReport", () => {
  it("formats critical items under Critical Issues heading", () => {
    const feedback: AggregatedFeedbackItem[] = [
      {
        severity: "critical",
        description: "SQL injection vulnerability in user input handler",
        affected_files: ["src/api.ts"],
        reasoning: "Unescaped user input flows into SQL query",
        dimension: "bugs",
        corroborated_by: ["openai", "google"],
        score: 5.5,
      },
    ];
    const report = formatReviewReport(feedback, {
      dimensions: ["bugs"],
      providers: ["openai", "google"],
      budget: "standard",
    });

    assert.ok(report.includes("## Critical Issues"), "should have Critical Issues heading");
    assert.ok(report.includes("SQL injection vulnerability"), "should include description");
    assert.ok(report.includes("src/api.ts"), "should include affected file");
    assert.ok(report.includes("Unescaped user input"), "should include reasoning");
  });

  it("formats warnings and suggestions in separate sections", () => {
    const feedback: AggregatedFeedbackItem[] = [
      {
        severity: "warning",
        description: "Missing error handling for network failures",
        affected_files: ["src/client.ts"],
        reasoning: "Network calls can fail",
        dimension: "completeness",
        corroborated_by: ["openai"],
        score: 2.5,
      },
      {
        severity: "suggestion",
        description: "Consider adding retry logic",
        affected_files: ["src/client.ts"],
        reasoning: "Improves reliability",
        dimension: "completeness",
        corroborated_by: ["google"],
        score: 1.5,
      },
    ];
    const report = formatReviewReport(feedback, {
      dimensions: ["completeness"],
      providers: ["openai", "google"],
      budget: "standard",
    });

    assert.ok(report.includes("## Warnings"), "should have Warnings heading");
    assert.ok(report.includes("## Suggestions"), "should have Suggestions heading");
    assert.ok(report.includes("Missing error handling"), "should include warning description");
    assert.ok(report.includes("Consider adding retry"), "should include suggestion description");
  });

  it("includes summary with dimension counts and provider names", () => {
    const feedback: AggregatedFeedbackItem[] = [
      {
        severity: "warning",
        description: "Issue 1",
        affected_files: [],
        reasoning: "Reason",
        dimension: "completeness",
        corroborated_by: ["openai"],
        score: 2.0,
      },
      {
        severity: "suggestion",
        description: "Issue 2",
        affected_files: [],
        reasoning: "Reason",
        dimension: "blind_spots",
        corroborated_by: ["google"],
        score: 1.0,
      },
    ];
    const report = formatReviewReport(feedback, {
      dimensions: ["completeness", "blind_spots"],
      providers: ["openai", "google"],
      budget: "standard",
    });

    assert.ok(report.includes("## Summary"), "should have Summary heading");
    assert.ok(report.includes("openai"), "should mention openai in summary");
    assert.ok(report.includes("google"), "should mention google in summary");
  });

  it("handles empty feedback with no-issues report", () => {
    const report = formatReviewReport([], {
      dimensions: ["completeness", "blind_spots"],
      providers: ["openai", "google"],
      budget: "standard",
    });

    assert.ok(
      report.includes("No issues found") || report.includes("no issues"),
      "should indicate no issues found",
    );
  });

  it("includes corroboration info showing which providers flagged each item", () => {
    const feedback: AggregatedFeedbackItem[] = [
      {
        severity: "warning",
        description: "Missing validation",
        affected_files: ["src/api.ts"],
        reasoning: "Input not validated",
        dimension: "completeness",
        corroborated_by: ["openai", "google"],
        score: 3.5,
      },
    ];
    const report = formatReviewReport(feedback, {
      dimensions: ["completeness"],
      providers: ["openai", "google"],
      budget: "standard",
    });

    assert.ok(
      report.includes("openai") && report.includes("google"),
      "should show corroborating providers",
    );
    assert.ok(
      report.includes("Flagged by") || report.includes("flagged by") || report.includes("Corroborated"),
      "should have corroboration label",
    );
  });

  it("omits section headings when no items of that severity exist", () => {
    const feedback: AggregatedFeedbackItem[] = [
      {
        severity: "suggestion",
        description: "Minor improvement",
        affected_files: [],
        reasoning: "Nice to have",
        dimension: "completeness",
        corroborated_by: ["openai"],
        score: 1.0,
      },
    ];
    const report = formatReviewReport(feedback, {
      dimensions: ["completeness"],
      providers: ["openai"],
      budget: "minimal",
    });

    assert.ok(!report.includes("## Critical Issues"), "should not have Critical Issues heading");
    assert.ok(!report.includes("## Warnings"), "should not have Warnings heading");
    assert.ok(report.includes("## Suggestions"), "should have Suggestions heading");
  });
});
