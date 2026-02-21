// ---------------------------------------------------------------------------
// Google provider — Gemini 2.0 Flash
// ---------------------------------------------------------------------------

import { GoogleGenerativeAI } from "@google/generative-ai";
import type { LLMProvider, ReviewRequest, ReviewResponse } from "./provider.js";
import type { FeedbackItem, ProviderName } from "../types.js";

const PROVIDER_NAME: ProviderName = "google";

function parseFeedback(raw: string, request: ReviewRequest): FeedbackItem[] {
  try {
    // Gemini may return markdown-wrapped JSON — strip fences
    const cleaned = raw.replace(/```json\n?/g, "").replace(/```\n?/g, "");
    const parsed = JSON.parse(cleaned);
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

export class GoogleProvider implements LLMProvider {
  readonly name = PROVIDER_NAME;
  readonly available: boolean;
  readonly models = ["gemini-2.0-flash"];
  private genAI: GoogleGenerativeAI | null;

  constructor() {
    const apiKey = process.env.GOOGLE_AI_API_KEY;
    this.available = Boolean(apiKey);
    this.genAI = apiKey ? new GoogleGenerativeAI(apiKey) : null;
  }

  async review(
    request: ReviewRequest,
    model = "gemini-2.0-flash",
  ): Promise<ReviewResponse> {
    if (!this.genAI) throw new Error("Google AI API key not configured");

    const genModel = this.genAI.getGenerativeModel({
      model,
      generationConfig: {
        temperature: 0.3,
        responseMimeType: "application/json",
      },
    });

    const result = await genModel.generateContent([
      { text: request.systemPrompt + "\n\n" + request.userPrompt },
    ]);

    const content = result.response.text();
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
