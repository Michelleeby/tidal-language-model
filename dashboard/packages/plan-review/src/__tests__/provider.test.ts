import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { ProviderRegistry } from "../providers/provider.js";
import type { LLMProvider, ReviewRequest, ReviewResponse } from "../providers/provider.js";
import type { ReviewDimension, ProviderName } from "../types.js";

// ---------------------------------------------------------------------------
// Mock provider factory
// ---------------------------------------------------------------------------

function createMockProvider(
  name: ProviderName,
  available: boolean,
  response?: Partial<ReviewResponse>,
): LLMProvider {
  return {
    name,
    available,
    models: ["mock-model-1"],
    async review(request: ReviewRequest): Promise<ReviewResponse> {
      return {
        feedback: response?.feedback ?? [
          {
            severity: "warning",
            description: `Mock feedback from ${name}`,
            affected_files: [],
            reasoning: "Mock reasoning",
            dimension: request.dimension,
            source: name,
          },
        ],
        provider: name,
        model: response?.model ?? "mock-model-1",
        dimension: request.dimension,
      };
    },
    estimateTokens(text: string): number {
      return Math.ceil(text.length / 4);
    },
  };
}

// ---------------------------------------------------------------------------
// ProviderRegistry
// ---------------------------------------------------------------------------

describe("ProviderRegistry", () => {
  it("registers and retrieves providers", () => {
    const registry = new ProviderRegistry();
    const provider = createMockProvider("openai", true);
    registry.register(provider);

    assert.equal(registry.get("openai"), provider);
  });

  it("returns undefined for unregistered provider", () => {
    const registry = new ProviderRegistry();
    assert.equal(registry.get("openai"), undefined);
  });

  it("lists only available providers", () => {
    const registry = new ProviderRegistry();
    registry.register(createMockProvider("openai", true));
    registry.register(createMockProvider("google", false));

    const available = registry.available();
    assert.equal(available.length, 1);
    assert.ok(available.some((p) => p.name === "openai"));
    assert.ok(!available.some((p) => p.name === "google"));
  });

  it("lists all registered providers regardless of availability", () => {
    const registry = new ProviderRegistry();
    registry.register(createMockProvider("openai", true));
    registry.register(createMockProvider("google", false));

    const all = registry.all();
    assert.equal(all.length, 2);
  });

  it("generates review from a mock provider", async () => {
    const registry = new ProviderRegistry();
    registry.register(createMockProvider("openai", true));

    const provider = registry.get("openai");
    assert.ok(provider);

    const response = await provider.review({
      systemPrompt: "You are a code reviewer",
      userPrompt: "Review this plan",
      dimension: "completeness" as ReviewDimension,
    });

    assert.equal(response.provider, "openai");
    assert.equal(response.dimension, "completeness");
    assert.ok(response.feedback.length > 0);
  });
});
