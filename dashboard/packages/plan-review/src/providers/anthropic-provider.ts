// ---------------------------------------------------------------------------
// Anthropic provider — Claude Sonnet
// ---------------------------------------------------------------------------

import Anthropic from "@anthropic-ai/sdk";
import type { LLMProvider, ReviewRequest, ReviewResponse } from "./provider.js";
import type { FeedbackItem, ProviderName } from "../types.js";

const PROVIDER_NAME: ProviderName = "anthropic";

function parseFeedback(raw: string, request: ReviewRequest): FeedbackItem[] {
  try {
    const parsed = JSON.parse(raw);
    const items: FeedbackItem[] = (parsed.feedback ?? []).map(
      (f: Record<string, unknown>) => ({
        severity: f.severity ?? "suggestion",
        description: String(f.description ?? ""),
        affected_files: Array.isArray(f.affected_files) ? f.affected_files : [],
        reasoning: String(f.reasoning ?? ""),
        dimension: request.dimension,
        source: PROVIDER_NAME,
      }),
    );
    return items;
  } catch {
    return [
      {
        severity: "suggestion",
        description: raw.slice(0, 500),
        affected_files: [],
        reasoning: "Raw response — could not parse structured output",
        dimension: request.dimension,
        source: PROVIDER_NAME,
      },
    ];
  }
}

export class AnthropicProvider implements LLMProvider {
  readonly name = PROVIDER_NAME;
  readonly available: boolean;
  readonly models = ["claude-sonnet-4-6"];
  private client: Anthropic | null;

  constructor() {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    this.available = Boolean(apiKey);
    this.client = apiKey ? new Anthropic({ apiKey }) : null;
  }

  async review(
    request: ReviewRequest,
    model = "claude-sonnet-4-6",
  ): Promise<ReviewResponse> {
    if (!this.client) throw new Error("Anthropic API key not configured");

    const response = await this.client.messages.create({
      model,
      max_tokens: 2048,
      temperature: 0.3,
      system:
        request.systemPrompt +
        '\n\nIMPORTANT: Respond ONLY with valid JSON in this format: {"feedback": [{"severity": "critical|warning|suggestion", "description": "...", "affected_files": ["..."], "reasoning": "..."}]}',
      messages: [{ role: "user", content: request.userPrompt }],
    });

    const content =
      response.content[0]?.type === "text" ? response.content[0].text : "{}";
    return {
      feedback: parseFeedback(content, request),
      provider: PROVIDER_NAME,
      model,
      dimension: request.dimension,
    };
  }

  estimateTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }
}
