import type { TrainingJob } from "@tidal/shared";
import type {
  ComputeProvider,
  ProvisionResult,
} from "../compute-provider.js";
import type { WorkerSpawner } from "../worker-spawner.js";

export class LocalProvider implements ComputeProvider {
  readonly type = "local" as const;
  readonly isRemote = false;

  constructor(private spawner: WorkerSpawner) {}

  async canProvision(): Promise<boolean> {
    return true;
  }

  async provision(_job: TrainingJob): Promise<ProvisionResult> {
    // Worker spawning is handled separately by WorkerSpawner
    return { success: true };
  }

  async deprovision(_job: TrainingJob): Promise<void> {
    // Process cleanup handled by WorkerSpawner
  }

  async isAlive(job: TrainingJob): Promise<boolean> {
    return this.spawner.isRunning(job.jobId);
  }
}
