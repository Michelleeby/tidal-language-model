// ── Compute & Job enums ──────────────────────────────────────────────

export type ComputeProviderType = "local" | "vastai" | "aws" | "digitalocean";

/** Job type is now a string determined by the plugin manifest (e.g. "lm-training", "rl-training"). */
export type JobType = string;

export type JobStatus =
  | "pending"
  | "provisioning"
  | "starting"
  | "running"
  | "completing"
  | "stopping"
  | "completed"
  | "failed"
  | "cancelled";

export type JobSignal = "complete" | "stop";

// ── Job record ───────────────────────────────────────────────────────

export interface JobConfig {
  type: JobType;
  plugin: string;
  configPath: string;
  resumeExpDir?: string;
  checkpoint?: string;
  rlConfigPath?: string;
  timesteps?: number;
}

export interface TrainingJob {
  jobId: string;
  type: JobType;
  status: JobStatus;
  provider: ComputeProviderType;
  config: JobConfig;
  experimentId?: string;
  createdAt: number;
  updatedAt: number;
  startedAt?: number;
  completedAt?: number;
  error?: string;
  providerMeta?: Record<string, unknown>;
}

// ── API request/response types ───────────────────────────────────────

export interface CreateJobRequest {
  type: JobType;
  plugin: string;
  configPath: string;
  provider?: ComputeProviderType;
  resumeExpDir?: string;
  checkpoint?: string;
  rlConfigPath?: string;
  timesteps?: number;
}

export interface CreateJobResponse {
  job: TrainingJob;
}

export interface JobsListResponse {
  jobs: TrainingJob[];
}

export interface JobResponse {
  job: TrainingJob | null;
}

export interface JobSignalRequest {
  signal: JobSignal;
}

export interface JobSignalResponse {
  ok: boolean;
  status: JobStatus;
}

// ── Log streaming ─────────────────────────────────────────────────────

export interface LogLine {
  timestamp: number;
  stream: "stdout" | "stderr";
  line: string;
}

export interface JobLogsResponse {
  jobId: string;
  lines: LogLine[];
  totalLines: number;
}
