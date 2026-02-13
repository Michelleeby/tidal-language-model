// ── Plugin manifest types ────────────────────────────────────────────
// These mirror the YAML structure of plugins/<name>/manifest.yaml.

export interface PluginManifest {
  name: string;
  displayName: string;
  version: string;
  description: string;
  trainingPhases: TrainingPhase[];
  checkpointPatterns: CheckpointPattern[];
  generation: GenerationConfig;
  metrics: MetricsConfig;
  redis: RedisConfig;
  infrastructure: InfraConfig;
}

export interface TrainingPhase {
  id: string;
  displayName: string;
  entrypoint: string;
  configFiles: string[];
  args: Record<string, string>;
  concurrency: number;
  gpuTier: string;
}

export interface CheckpointPattern {
  phase: string;
  glob: string;
  epochCapture?: string;
  excludePrefix?: string;
}

export interface GenerationConfig {
  entrypoint: string;
  args: Record<string, string>;
  defaultConfigPath: string;
  modes: GenerationMode[];
  parameters: GenerationParameter[];
  modelCheckpointPatterns: string[];
  rlCheckpointPatterns: string[];
}

export interface GenerationMode {
  id: string;
  displayName: string;
  requiresRLCheckpoint: boolean;
}

export interface GenerationParameter {
  id: string;
  displayName: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface MetricsConfig {
  redisPrefix: string;
  lm: LMMetricsConfig;
  rl: RLMetricsConfig;
}

export interface LMMetricsConfig {
  directory: string;
  historyFile: string;
  statusFile: string;
  latestFile: string;
  primaryKeys: string[];
}

export interface RLMetricsConfig {
  directory: string;
  metricsFile: string;
  primaryKeys: string[];
}

export interface RedisConfig {
  jobsHash: string;
  jobsActiveSet: string;
  signalPrefix: string;
  heartbeatPrefix: string;
  updatesChannel: string;
  experimentsSet: string;
}

export interface InfraConfig {
  pythonEnv: string;
  dockerImage: string;
  requirementsFile: string;
  gpuTiers: Record<string, GpuTierSpec>;
}

export interface GpuTierSpec {
  minGpuRamMb: number;
  minCpuCores: number;
}

// ── API response types for plugin endpoints ─────────────────────────

export interface PluginSummary {
  name: string;
  displayName: string;
  version: string;
  trainingPhases: Array<{ id: string; displayName: string }>;
  generationModes: Array<{ id: string; displayName: string }>;
}

export interface PluginsListResponse {
  plugins: PluginSummary[];
}

export interface PluginResponse {
  plugin: PluginManifest | null;
}
