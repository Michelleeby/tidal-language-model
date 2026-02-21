import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { ProviderRegistry } from "../providers/provider.js";
import type { LLMProvider, ReviewRequest, ReviewResponse } from "../providers/provider.js";
import type { ReviewDimension, ProviderName, FeedbackItem } from "../types.js";
import {
  handleReviewPlan,
  handleReviewDimension,
  handleSummarizeADRs,
  handleAggregateFeedback,
  handleListProviders,
  handleGetReviewCosts,
} from "../tools/review-tools.js";

// ---------------------------------------------------------------------------
// Mock provider factory
// ---------------------------------------------------------------------------

function createMockProvider(
  name: ProviderName,
  available: boolean,
  feedback?: FeedbackItem[],
): LLMProvider {
  return {
    name,
    available,
    models: ["mock-model"],
    async review(request: ReviewRequest): Promise<ReviewResponse> {
      return {
        feedback: feedback ?? [
          {
            severity: "warning",
            description: `Feedback from ${name}: missing error handling`,
            affected_files: ["src/main.ts"],
            reasoning: `${name} found this issue`,
            dimension: request.dimension,
            source: name,
          },
        ],
        provider: name,
        model: "mock-model",
        dimension: request.dimension,
      };
    },
    estimateTokens(text: string): number {
      return Math.ceil(text.length / 4);
    },
  };
}

function makeRegistry(...providers: LLMProvider[]): ProviderRegistry {
  const registry = new ProviderRegistry();
  for (const p of providers) registry.register(p);
  return registry;
}

// ---------------------------------------------------------------------------
// handleListProviders
// ---------------------------------------------------------------------------

describe("handleListProviders", () => {
  it("returns available and unavailable providers", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", false),
      createMockProvider("anthropic", true),
    );
    const result = await handleListProviders(registry);
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.available.length, 2);
    assert.equal(parsed.unavailable.length, 1);
    assert.ok(parsed.available.includes("openai"));
    assert.ok(parsed.unavailable.includes("google"));
  });
});

// ---------------------------------------------------------------------------
// handleGetReviewCosts
// ---------------------------------------------------------------------------

describe("handleGetReviewCosts", () => {
  it("returns cost estimates for a plan", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", true),
    );
    const result = await handleGetReviewCosts(registry, {
      plan: "A short test plan",
      budget: "minimal",
    });
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    assert.ok(parsed.estimates);
    assert.ok(parsed.totalEstimatedCostUsd >= 0);
  });
});

// ---------------------------------------------------------------------------
// handleReviewDimension
// ---------------------------------------------------------------------------

describe("handleReviewDimension", () => {
  it("returns feedback for a single dimension from a single provider", async () => {
    const registry = makeRegistry(createMockProvider("openai", true));
    const result = await handleReviewDimension(registry, {
      plan: "Add a new API endpoint for user profiles",
      dimension: "completeness",
      provider: "openai",
    });
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    assert.ok(parsed.feedback.length > 0);
    assert.equal(parsed.provider, "openai");
    assert.equal(parsed.dimension, "completeness");
  });

  it("returns error for unavailable provider", async () => {
    const registry = makeRegistry(createMockProvider("openai", false));
    const result = await handleReviewDimension(registry, {
      plan: "Some plan",
      dimension: "completeness",
      provider: "openai",
    });
    assert.equal(result.isError, true);
  });
});

// ---------------------------------------------------------------------------
// handleReviewPlan
// ---------------------------------------------------------------------------

describe("handleReviewPlan", () => {
  it("orchestrates review across dimensions and providers", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", true),
    );
    const result = await handleReviewPlan(registry, {
      plan: "Implement a new feature for the gating controller",
      adrDir: null,
    });
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    assert.ok(parsed.feedback.length > 0);
    assert.ok(parsed.dimensions_reviewed);
    assert.ok(parsed.providers_used);
  });

  it("uses specified dimensions when provided", async () => {
    const registry = makeRegistry(createMockProvider("openai", true));
    const result = await handleReviewPlan(registry, {
      plan: "Some plan",
      dimensions: ["completeness", "test_coverage"],
      adrDir: null,
    });
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    assert.deepEqual(
      parsed.dimensions_reviewed.sort(),
      ["completeness", "test_coverage"].sort(),
    );
  });

  it("uses specified providers when provided", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", true),
    );
    const result = await handleReviewPlan(registry, {
      plan: "Some plan",
      providers: ["openai"],
      adrDir: null,
    });
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    assert.deepEqual(parsed.providers_used, ["openai"]);
  });

  it("returns error when no providers are available", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", false),
      createMockProvider("google", false),
    );
    const result = await handleReviewPlan(registry, {
      plan: "Some plan",
      adrDir: null,
    });
    assert.equal(result.isError, true);
  });
});

// ---------------------------------------------------------------------------
// handleSummarizeADRs
// ---------------------------------------------------------------------------

describe("handleSummarizeADRs", () => {
  it("returns error when adrDir is null", async () => {
    const result = await handleSummarizeADRs({ adrDir: null });
    assert.equal(result.isError, true);
  });
});

// ---------------------------------------------------------------------------
// handleAggregateFeedback
// ---------------------------------------------------------------------------

describe("handleAggregateFeedback", () => {
  it("deduplicates and ranks raw feedback items", async () => {
    const items: FeedbackItem[] = [
      {
        severity: "warning",
        description: "Missing error handling for network failures",
        affected_files: ["api.ts"],
        reasoning: "Network calls fail",
        dimension: "completeness",
        source: "openai",
      },
      {
        severity: "warning",
        description: "No error handling for network failure scenarios",
        affected_files: ["api.ts"],
        reasoning: "HTTP can fail",
        dimension: "completeness",
        source: "google",
      },
      {
        severity: "critical",
        description: "Checkpoint format change breaks loading",
        affected_files: ["checkpoint.py"],
        reasoning: "Incompatible format",
        dimension: "regression_risk",
        source: "anthropic",
      },
    ];
    const result = await handleAggregateFeedback({ feedback: items });
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    // Should have 2 items (first two merged)
    assert.equal(parsed.feedback.length, 2);
    // Sorted by score
    assert.ok(parsed.feedback[0].score >= parsed.feedback[1].score);
  });

  it("handles empty feedback array", async () => {
    const result = await handleAggregateFeedback({ feedback: [] });
    assert.equal(result.isError, undefined);
    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.feedback.length, 0);
  });
});
