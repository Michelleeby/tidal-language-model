// ---------------------------------------------------------------------------
// OpenAI provider — GPT-4o, o3-mini
// ---------------------------------------------------------------------------

import OpenAI from "openai";
import type { LLMProvider, ReviewRequest, ReviewResponse } from "./provider.js";
import type { FeedbackItem, ProviderName } from "../types.js";

const PROVIDER_NAME: ProviderName = "openai";

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

export class OpenAIProvider implements LLMProvider {
  readonly name = PROVIDER_NAME;
  readonly available: boolean;
  readonly models = ["gpt-4o", "o3-mini"];
  private client: OpenAI | null;

  constructor() {
    const apiKey = process.env.OPENAI_API_KEY;
    this.available = Boolean(apiKey);
    this.client = apiKey ? new OpenAI({ apiKey }) : null;
  }

  async review(
    request: ReviewRequest,
    model = "gpt-4o",
  ): Promise<ReviewResponse> {
    if (!this.client) throw new Error("OpenAI API key not configured");

    const response = await this.client.chat.completions.create({
      model,
      temperature: 0.3,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: request.systemPrompt },
        { role: "user", content: request.userPrompt },
      ],
    });

    const content = response.choices[0]?.message?.content ?? "{}";
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
