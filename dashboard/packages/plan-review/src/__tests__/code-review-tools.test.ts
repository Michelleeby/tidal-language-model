import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { ProviderRegistry } from "../providers/provider.js";
import type { LLMProvider, ReviewRequest, ReviewResponse } from "../providers/provider.js";
import type { ProviderName, FeedbackItem } from "../types.js";
import { handleReviewCode } from "../tools/code-review-tools.js";

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
            description: `Code feedback from ${name}/${model ?? "default"}: potential bug`,
            affected_files: ["src/main.py"],
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
// handleReviewCode
// ---------------------------------------------------------------------------

describe("handleReviewCode", () => {
  it("returns feedback for a diff across all 3 dimensions", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", true),
    );
    const result = await handleReviewCode(registry, {
      diff: "diff --git a/src/main.py b/src/main.py\n+some code change",
      adrDir: null,
    });
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    assert.ok(parsed.feedback.length > 0);
    assert.ok(parsed.dimensions_reviewed);
    assert.equal(parsed.dimensions_reviewed.length, 3);
    assert.ok(parsed.providers_used);
  });

  it("uses specified dimensions when provided", async () => {
    const registry = makeRegistry(createMockProvider("openai", true));
    const result = await handleReviewCode(registry, {
      diff: "some diff",
      dimensions: ["bugs"],
      adrDir: null,
    });
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    assert.deepEqual(parsed.dimensions_reviewed, ["bugs"]);
  });

  it("returns error when no providers available", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", false),
      createMockProvider("google", false),
    );
    const result = await handleReviewCode(registry, {
      diff: "some diff",
      adrDir: null,
    });
    assert.equal(result.isError, true);
  });

  it("routes bugs to openai at standard budget", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", true),
    );
    const result = await handleReviewCode(registry, {
      diff: "some diff",
      dimensions: ["bugs"],
      budget: "standard",
      adrDir: null,
    });
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    assert.ok(parsed.providers_used.includes("openai"));
  });

  it("routes adr_compliance to google at standard budget", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", true),
    );
    const result = await handleReviewCode(registry, {
      diff: "some diff",
      dimensions: ["adr_compliance"],
      budget: "standard",
      adrDir: null,
    });
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    assert.ok(parsed.providers_used.includes("google"));
  });

  it("routes hypothesis_alignment to openai and google at standard budget", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", true),
    );
    const result = await handleReviewCode(registry, {
      diff: "some diff",
      dimensions: ["hypothesis_alignment"],
      budget: "standard",
      adrDir: null,
    });
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    assert.ok(parsed.providers_used.includes("openai"));
    assert.ok(parsed.providers_used.includes("google"));
  });

  it("uses single provider per dimension at minimal budget", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", true),
    );
    const result = await handleReviewCode(registry, {
      diff: "some diff",
      dimensions: ["bugs"],
      budget: "minimal",
      adrDir: null,
    });
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    // minimal = 1 provider per dimension
    assert.equal(parsed.providers_used.length, 1);
  });

  it("thorough budget makes 3 model calls per dimension", async () => {
    // Track calls to verify 3 distinct calls are made
    const calls: Array<{ provider: string; model?: string }> = [];
    const openaiProvider: LLMProvider = {
      name: "openai",
      available: true,
      models: ["gpt-4o", "o3-mini"],
      async review(request: ReviewRequest, model?: string): Promise<ReviewResponse> {
        calls.push({ provider: "openai", model });
        return {
          feedback: [{
            severity: "warning",
            description: `OpenAI ${model} feedback`,
            affected_files: [],
            reasoning: "test",
            dimension: request.dimension,
            source: "openai",
          }],
          provider: "openai",
          model: model ?? "gpt-4o",
          dimension: request.dimension,
        };
      },
      estimateTokens: (t) => Math.ceil(t.length / 4),
    };
    const googleProvider: LLMProvider = {
      name: "google",
      available: true,
      models: ["gemini-2.0-flash"],
      async review(request: ReviewRequest, model?: string): Promise<ReviewResponse> {
        calls.push({ provider: "google", model });
        return {
          feedback: [{
            severity: "suggestion",
            description: `Google ${model} feedback`,
            affected_files: [],
            reasoning: "test",
            dimension: request.dimension,
            source: "google",
          }],
          provider: "google",
          model: model ?? "gemini-2.0-flash",
          dimension: request.dimension,
        };
      },
      estimateTokens: (t) => Math.ceil(t.length / 4),
    };

    const registry = makeRegistry(openaiProvider, googleProvider);
    const result = await handleReviewCode(registry, {
      diff: "some diff",
      dimensions: ["bugs"],
      budget: "thorough",
      adrDir: null,
    });
    assert.equal(result.isError, undefined);

    // Should have made 3 calls for 1 dimension
    assert.equal(calls.length, 3, "thorough should make 3 calls per dimension");

    // Verify 2 distinct providers used
    const parsed = JSON.parse(result.content[0].text as string);
    assert.ok(parsed.providers_used.includes("openai"));
    assert.ok(parsed.providers_used.includes("google"));
  });
});
