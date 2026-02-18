import type {
  MetricPoint,
  TrainingStatus,
  RLTrainingMetrics,
  AblationResults,
  EvaluationResults,
  ExperimentSummary,
  CheckpointInfo,
} from "./metrics.js";

export type { GpuInstanceResponse } from "./gpu-instance.js";

/** GET /api/experiments */
export interface ExperimentsResponse {
  experiments: ExperimentSummary[];
}

/** GET /api/experiments/:expId/metrics query params */
export interface MetricsQuery {
  mode: "recent" | "historical";
  window?: number;
  maxPoints?: number;
}

/** GET /api/experiments/:expId/metrics */
export interface MetricsResponse {
  expId: string;
  points: MetricPoint[];
  totalPoints: number;
  originalCount: number;
  downsampled: boolean;
}

/** GET /api/experiments/:expId/rl-metrics */
export interface RLMetricsResponse {
  expId: string;
  metrics: RLTrainingMetrics | null;
}

/** GET /api/experiments/:expId/status */
export interface StatusResponse {
  expId: string;
  status: TrainingStatus | null;
}

/** GET /api/experiments/:expId/checkpoints */
export interface CheckpointsResponse {
  expId: string;
  checkpoints: CheckpointInfo[];
}

/** GET /api/experiments/:expId/evaluation */
export interface EvaluationResponse {
  expId: string;
  results: EvaluationResults | null;
}

/** GET /api/experiments/:expId/ablation */
export interface AblationResponse {
  expId: string;
  results: AblationResults | null;
}

/** POST /api/generate */
export interface GenerateRequest {
  checkpoint: string;
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  gatingMode?: "none" | "random" | "fixed" | "learned";
  rlCheckpoint?: string;
  modulation?: number;
}

export interface GateEffectsStep {
  temperature: number;
  repetition_penalty: number;
  top_k: number;
  top_p: number;
}

export interface GenerationTrajectory {
  gateSignals: number[][];
  effects: GateEffectsStep[];
  tokenIds: number[];
  tokenTexts: string[];
}

export interface GenerateResponse {
  text: string;
  tokensGenerated: number;
  elapsedMs: number;
  trajectory?: GenerationTrajectory;
}

/** POST /api/analyze-trajectories */
export interface AnalyzeRequest {
  checkpoint: string;
  prompts: string[];
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  samplesPerPrompt?: number;
  gatingMode?: "random" | "fixed" | "learned";
  rlCheckpoint?: string;
  includeExtremeValues?: boolean;
  bootstrap?: boolean;
}

export interface SignalStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  q25: number;
  q50: number;
  q75: number;
}

export interface BatchAnalysis {
  perPromptSummaries: Record<string, unknown>;
  crossPromptVariance: Record<string, { betweenPromptVar: number; withinPromptVar: number }>;
  strategyCharacterization: Record<string, { globalMean: number; globalStd: number }>;
}

export interface SweepAnalysis {
  configComparisons: Record<string, {
    signalStats: Record<string, SignalStats>;
    textProperties: { wordCount: number; uniqueTokenRatio: number; charCount: number };
  }>;
  interpretabilityMap: Record<string, {
    lowConfig: string;
    highConfig: string;
    effect: Record<string, { low: number; high: number; delta: number }>;
  }>;
}

export interface AnalyzeResponse {
  batchAnalysis: BatchAnalysis;
  trajectories: Record<string, GenerationTrajectory[]>;
  sweepAnalysis?: SweepAnalysis;
  sweepTexts?: Record<string, string>;
}

/**
 * Curated TinyStories prompts across 7 narrative categories.
 * Shared so both MCP tools and client report blocks use the same set.
 */
export const CURATED_PROMPTS: string[] = [
  // Fairy tale (3)
  "Once upon a time, in a land far away,",
  "There was a tiny dragon who could not breathe fire.",
  "The princess did not want to be rescued.",
  // Character intro (3)
  "Lily was a little girl who loved to paint.",
  "Tom was a boy who never stopped asking questions.",
  "The old woman at the end of the street had a secret.",
  // Action (3)
  "The rabbit ran as fast as it could through the forest.",
  "Suddenly, a loud noise came from the kitchen.",
  "The boat rocked back and forth on the stormy sea.",
  // Emotional (3)
  "Sam felt sad because his best friend moved away.",
  "The little bird was afraid to fly for the first time.",
  "Mia was so happy she could not stop smiling.",
  // Discovery (3)
  "Behind the old bookshelf, there was a hidden door.",
  "The children found a strange map in the attic.",
  "Inside the box was something nobody expected.",
  // Everyday (3)
  "It was a sunny morning and the birds were singing.",
  "Mom said it was time to go to the store.",
  "The dog waited by the door every afternoon.",
  // Dialogue (2)
  "\"Can you help me?\" asked the small kitten.",
  "\"I have an idea,\" said Ben with a big smile.",
];

/** GET /api/plugins/:name/configs/:filename */
export interface ConfigFileResponse {
  filename: string;
  content: string;
}

/** GET /api/plugins/:name/configs */
export interface ConfigListResponse {
  plugin: string;
  files: string[];
}
