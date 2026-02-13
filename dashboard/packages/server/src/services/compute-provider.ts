import type { ComputeProviderType, TrainingJob } from "@tidal/shared";
import type { GpuTier } from "./job-policy.js";

export interface ProvisionResult {
  success: boolean;
  meta?: Record<string, unknown>;
  error?: string;
}

export interface ComputeProvider {
  readonly type: ComputeProviderType;
  readonly isRemote: boolean;
  canProvision(): Promise<boolean>;
  provision(job: TrainingJob, gpuTier?: GpuTier): Promise<ProvisionResult>;
  deprovision(job: TrainingJob): Promise<void>;
  isAlive(job: TrainingJob): Promise<boolean>;
}
