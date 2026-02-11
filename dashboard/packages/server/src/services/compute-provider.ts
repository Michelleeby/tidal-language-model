import type { ComputeProviderType, TrainingJob } from "@tidal/shared";

export interface ProvisionResult {
  success: boolean;
  meta?: Record<string, unknown>;
  error?: string;
}

export interface ComputeProvider {
  readonly type: ComputeProviderType;
  canProvision(): Promise<boolean>;
  provision(job: TrainingJob): Promise<ProvisionResult>;
  deprovision(job: TrainingJob): Promise<void>;
  isAlive(job: TrainingJob): Promise<boolean>;
}
