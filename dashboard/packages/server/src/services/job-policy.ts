import type { JobType, TrainingJob, PluginManifest } from "@tidal/shared";

export type GpuTier = string;

export interface JobPolicy {
  readonly type: JobType;
  /** Return null if OK to proceed, or an error message string if blocked. */
  checkConcurrency(activeJobs: TrainingJob[]): string | null;
  /** GPU tier to request when provisioning remote instances. */
  gpuTier(): GpuTier;
}

/**
 * A generic job policy driven by manifest training phase config.
 * Replaces the old LMTrainingPolicy / RLTrainingPolicy classes.
 */
export class ManifestJobPolicy implements JobPolicy {
  readonly type: JobType;

  constructor(
    type: JobType,
    private displayName: string,
    private concurrency: number,
    private tier: string,
  ) {
    this.type = type;
  }

  checkConcurrency(activeJobs: TrainingJob[]): string | null {
    const sameType = activeJobs.filter((j) => j.type === this.type);
    if (sameType.length >= this.concurrency) {
      return `A ${this.displayName} job is already running (max ${this.concurrency})`;
    }
    return null;
  }

  gpuTier(): GpuTier {
    return this.tier;
  }
}

export class JobPolicyRegistry {
  private policies: Map<JobType, JobPolicy>;

  constructor(manifest?: PluginManifest | null) {
    this.policies = new Map();

    if (manifest) {
      for (const phase of manifest.trainingPhases) {
        this.register(
          new ManifestJobPolicy(
            phase.id,
            phase.displayName,
            phase.concurrency,
            phase.gpuTier,
          ),
        );
      }
    }
  }

  register(policy: JobPolicy): void {
    this.policies.set(policy.type, policy);
  }

  get(type: JobType): JobPolicy | undefined {
    return this.policies.get(type);
  }
}
