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
  assignProviders,
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
    models: name === "openai" ? ["gpt-4o", "o3-mini"] : ["gemini-2.0-flash"],
    async review(request: ReviewRequest, model?: string): Promise<ReviewResponse> {
      return {
        feedback: feedback ?? [
          {
            severity: "warning",
            description: `Feedback from ${name}/${model ?? "default"}: missing error handling`,
            affected_files: ["src/main.ts"],
            reasoning: `${name} found this issue`,
            dimension: request.dimension,
            source: name,
          },
        ],
        provider: name,
        model: model ?? "mock-model",
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
// assignProviders — direct unit tests
// ---------------------------------------------------------------------------

describe("assignProviders", () => {
  it("standard budget maps all dimensions to openai and google only", () => {
    const assignments = assignProviders(
      ["completeness", "blind_spots", "regression_risk", "test_coverage", "hypothesis_scope"],
      ["openai", "google"],
      "standard",
    );

    for (const a of assignments) {
      const providers = a.assignments.map((m) => m.provider);
      assert.ok(providers.includes("openai") || providers.includes("google"),
        `${a.dimension} should use openai or google`);
      for (const ma of a.assignments) {
        assert.ok(
          ma.provider === "openai" || ma.provider === "google",
          `${a.dimension} has unexpected provider: ${ma.provider}`,
        );
      }
      assert.equal(a.assignments.length, 2, `${a.dimension} should have 2 assignments at standard`);
    }
  });

  it("thorough budget returns 3 model assignments with specific models", () => {
    const assignments = assignProviders(
      ["completeness"],
      ["openai", "google"],
      "thorough",
    );

    assert.equal(assignments.length, 1);
    const a = assignments[0];
    assert.equal(a.assignments.length, 3, "thorough should have 3 model calls");

    // Verify the specific model assignments
    const models = a.assignments.map((m) => `${m.provider}/${m.model ?? "default"}`);
    assert.ok(models.includes("openai/gpt-4o"), "should include openai/gpt-4o");
    assert.ok(models.includes("openai/o3-mini"), "should include openai/o3-mini");
    assert.ok(
      a.assignments.some((m) => m.provider === "google"),
      "should include google",
    );
  });

  it("minimal budget uses 1 call per dimension", () => {
    const assignments = assignProviders(
      ["completeness", "blind_spots", "regression_risk", "test_coverage", "hypothesis_scope"],
      ["openai", "google"],
      "minimal",
    );

    for (const a of assignments) {
      assert.equal(a.assignments.length, 1, `${a.dimension} should have 1 assignment at minimal`);
    }
  });

  it("minimal budget alternates between openai and google", () => {
    const assignments = assignProviders(
      ["completeness", "blind_spots", "regression_risk", "test_coverage", "hypothesis_scope"],
      ["openai", "google"],
      "minimal",
    );

    const providers = assignments.map((a) => a.assignments[0].provider);
    assert.ok(providers.includes("openai"), "should use openai for some dimensions");
    assert.ok(providers.includes("google"), "should use google for some dimensions");
  });

  it("falls back to available providers when preferred is missing", () => {
    const assignments = assignProviders(
      ["completeness"],
      ["google"],
      "standard",
    );

    assert.equal(assignments.length, 1);
    assert.ok(assignments[0].assignments.length > 0, "should have at least one assignment");
    assert.equal(assignments[0].assignments[0].provider, "google");
  });
});

// ---------------------------------------------------------------------------
// handleListProviders
// ---------------------------------------------------------------------------

describe("handleListProviders", () => {
  it("returns available and unavailable providers", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", false),
    );
    const result = await handleListProviders(registry);
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.available.length, 1);
    assert.equal(parsed.unavailable.length, 1);
    assert.ok(parsed.available.includes("openai"));
    assert.ok(parsed.unavailable.includes("google"));
  });

  it("does not include anthropic in any list", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", true),
    );
    const result = await handleListProviders(registry);
    const parsed = JSON.parse(result.content[0].text as string);

    const all = [...parsed.available, ...parsed.unavailable];
    assert.ok(!all.includes("anthropic"), "anthropic should not appear");
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

  it("has no anthropic cost line", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", true),
    );
    const result = await handleGetReviewCosts(registry, {
      plan: "A short test plan",
      budget: "thorough",
    });
    const parsed = JSON.parse(result.content[0].text as string);

    for (const estimate of parsed.estimates) {
      assert.notEqual(estimate.provider, "anthropic", "no anthropic cost line");
    }
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
        source: "google",
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
