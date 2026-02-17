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
}

export interface GateEffectsStep {
  temperature: number;
  repetition_penalty: number;
  top_k: number;
  top_p: number;
}

export interface GenerationTrajectory {
  gateSignals: [number, number, number][];
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
