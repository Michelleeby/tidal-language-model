import type { TrainingJob } from "@tidal/shared";
import type {
  ComputeProvider,
  ProvisionResult,
} from "../compute-provider.js";

export class AWSProvider implements ComputeProvider {
  readonly type = "aws" as const;
  readonly isRemote = true;

  async canProvision(): Promise<boolean> {
    return false;
  }

  async provision(_job: TrainingJob): Promise<ProvisionResult> {
    return { success: false, error: "AWS provider not implemented" };
  }

  async deprovision(_job: TrainingJob): Promise<void> {}

  async isAlive(_job: TrainingJob): Promise<boolean> {
    return false;
  }
}
