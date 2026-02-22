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
    models: ["mock-model"],
    async review(request: ReviewRequest): Promise<ReviewResponse> {
      return {
        feedback: feedback ?? [
          {
            severity: "warning",
            description: `Code feedback from ${name}: potential bug`,
            affected_files: ["src/main.py"],
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
// handleReviewCode
// ---------------------------------------------------------------------------

describe("handleReviewCode", () => {
  it("returns feedback for a diff across all 3 dimensions", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", true),
      createMockProvider("anthropic", true),
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

  it("routes hypothesis_alignment to openai and anthropic at standard budget", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("anthropic", true),
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
    assert.ok(parsed.providers_used.includes("anthropic"));
  });

  it("uses single provider per dimension at minimal budget", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", true),
      createMockProvider("anthropic", true),
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

  it("uses all providers per dimension at thorough budget", async () => {
    const registry = makeRegistry(
      createMockProvider("openai", true),
      createMockProvider("google", true),
      createMockProvider("anthropic", true),
    );
    const result = await handleReviewCode(registry, {
      diff: "some diff",
      dimensions: ["bugs"],
      budget: "thorough",
      adrDir: null,
    });
    assert.equal(result.isError, undefined);

    const parsed = JSON.parse(result.content[0].text as string);
    // thorough = all 3 providers
    assert.equal(parsed.providers_used.length, 3);
  });
});
