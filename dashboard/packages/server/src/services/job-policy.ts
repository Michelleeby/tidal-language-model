import type { JobType, TrainingJob } from "@tidal/shared";

export type GpuTier = "standard";

export interface JobPolicy {
  readonly type: JobType;
  /** Return null if OK to proceed, or an error message string if blocked. */
  checkConcurrency(activeJobs: TrainingJob[]): string | null;
  /** GPU tier to request when provisioning remote instances. */
  gpuTier(): GpuTier;
}

export class LMTrainingPolicy implements JobPolicy {
  readonly type: JobType = "lm-training";

  checkConcurrency(activeJobs: TrainingJob[]): string | null {
    const existing = activeJobs.find((j) => j.type === "lm-training");
    return existing
      ? "An LM training job is already running"
      : null;
  }

  gpuTier(): GpuTier {
    return "standard";
  }
}

export class RLTrainingPolicy implements JobPolicy {
  readonly type: JobType = "rl-training";

  checkConcurrency(activeJobs: TrainingJob[]): string | null {
    const existing = activeJobs.find((j) => j.type === "rl-training");
    return existing
      ? "An RL training job is already running"
      : null;
  }

  gpuTier(): GpuTier {
    return "standard";
  }
}

export class JobPolicyRegistry {
  private policies: Map<JobType, JobPolicy>;

  constructor() {
    this.policies = new Map();
    this.register(new LMTrainingPolicy());
    this.register(new RLTrainingPolicy());
  }

  register(policy: JobPolicy): void {
    this.policies.set(policy.type, policy);
  }

  get(type: JobType): JobPolicy | undefined {
    return this.policies.get(type);
  }
}
