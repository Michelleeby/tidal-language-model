// ---------------------------------------------------------------------------
// Shared types for the plan-review MCP server
// ---------------------------------------------------------------------------

export type Severity = "critical" | "warning" | "suggestion";

export type ReviewDimension =
  | "completeness"
  | "blind_spots"
  | "regression_risk"
  | "test_coverage"
  | "hypothesis_scope";

export type ProviderName = "openai" | "google" | "anthropic";

export type Budget = "minimal" | "standard" | "thorough";

export interface FeedbackItem {
  severity: Severity;
  description: string;
  affected_files: string[];
  reasoning: string;
  dimension: ReviewDimension;
  source: ProviderName;
}

export interface AggregatedFeedbackItem {
  severity: Severity;
  description: string;
  affected_files: string[];
  reasoning: string;
  dimension: ReviewDimension;
  corroborated_by: ProviderName[];
  score: number;
}

export interface ADRSummary {
  number: number;
  title: string;
  status: string;
  context: string;
  decision: string;
  keywords: string[];
  files_affected: string[];
  raw: string;
}

export interface ReviewRequest {
  systemPrompt: string;
  userPrompt: string;
  dimension: ReviewDimension;
}

export interface ReviewResponse {
  feedback: FeedbackItem[];
  provider: ProviderName;
  model: string;
  dimension: ReviewDimension;
}

export interface CostEstimate {
  provider: ProviderName;
  model: string;
  estimatedInputTokens: number;
  estimatedOutputTokens: number;
  estimatedCostUsd: number;
}
