// ---------------------------------------------------------------------------
// LLM Provider abstraction + registry
// ---------------------------------------------------------------------------

import type { FeedbackItem, AnyDimension, ProviderName } from "../types.js";

export interface ReviewRequest {
  systemPrompt: string;
  userPrompt: string;
  dimension: AnyDimension;
}

export interface ReviewResponse {
  feedback: FeedbackItem[];
  provider: ProviderName;
  model: string;
  dimension: AnyDimension;
}

export interface LLMProvider {
  readonly name: ProviderName;
  readonly available: boolean;
  readonly models: string[];
  review(request: ReviewRequest, model?: string): Promise<ReviewResponse>;
  estimateTokens(text: string): number;
}

export class ProviderRegistry {
  private providers = new Map<ProviderName, LLMProvider>();

  register(provider: LLMProvider): void {
    this.providers.set(provider.name, provider);
  }

  get(name: ProviderName): LLMProvider | undefined {
    return this.providers.get(name);
  }

  /** Only providers with valid API keys. */
  available(): LLMProvider[] {
    return [...this.providers.values()].filter((p) => p.available);
  }

  /** All registered providers regardless of availability. */
  all(): LLMProvider[] {
    return [...this.providers.values()];
  }
}
