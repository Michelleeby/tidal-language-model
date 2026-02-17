/** A single training metric data point (matches MetricsLogger.log_metrics output). */
export interface MetricPoint {
  step: number;
  timestamp: number;
  "Losses/Total"?: number;
  "Learning Rate"?: number;
  [key: string]: number | string | undefined;
}

/** Training status (matches MetricsLogger._update_status output). */
export interface TrainingStatus {
  status: "initialized" | "training" | "completed";
  start_time?: number;
  end_time?: number;
  last_update: number;
  current_step?: number;
  total_metrics_logged?: number;
  total_steps?: number;
}

/** RL training history (matches RLTrainer._save_metrics output). */
export interface RLTrainingHistory {
  episode_rewards: number[];
  policy_loss: number[];
  value_loss: number[];
  entropy: number[];
  gate_creativity?: number[];
  gate_focus?: number[];
  gate_stability?: number[];
  reward_perplexity?: number[];
  reward_diversity?: number[];
  reward_focus?: number[];
  reward_repetition?: number[];
  reward_coherence?: number[];
  explained_variance?: number[];
}

/** RL metrics snapshot from Redis / disk. */
export interface RLTrainingMetrics {
  global_step: number;
  timestamp: number;
  history: RLTrainingHistory;
}

/** Ablation study results (matches ablation_results.json). */
export interface AblationPolicyResult {
  mean_reward: number;
  std_reward: number;
  mean_diversity: number;
  mean_perplexity: number;
}

export type AblationResults = Record<string, AblationPolicyResult>;

/** Evaluation results (from Evaluator). */
export interface EvaluationResults {
  perplexity: number;
  samples: GeneratedSample[];
  timestamp?: string;
}

export interface GeneratedSample {
  prompt: string;
  generated: string;
  temperature?: number;
  top_k?: number;
}

/** Experiment summary for listing. */
export interface ExperimentSummary {
  id: string;
  path: string;
  created: number;
  hasRLMetrics: boolean;
  hasEvaluation: boolean;
  hasAblation: boolean;
  hasGpuInstance: boolean;
  status: TrainingStatus | null;
  checkpoints: string[];
}

/** Checkpoint file info. */
export interface CheckpointInfo {
  filename: string;
  path: string;
  sizeBytes: number;
  modified: number;
  phase: string;
  epoch?: number;
}
