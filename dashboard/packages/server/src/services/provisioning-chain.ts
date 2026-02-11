import type { ComputeProviderType, TrainingJob } from "@tidal/shared";
import type { ComputeProvider, ProvisionResult } from "./compute-provider.js";

export class ProvisioningChain {
  private providers: ComputeProvider[];
  private byType: Map<ComputeProviderType, ComputeProvider>;

  constructor(providers: ComputeProvider[]) {
    this.providers = providers;
    this.byType = new Map(providers.map((p) => [p.type, p]));
  }

  getProvider(type: ComputeProviderType): ComputeProvider | undefined {
    return this.byType.get(type);
  }

  async provision(job: TrainingJob): Promise<ProvisionResult> {
    for (const provider of this.providers) {
      if (await provider.canProvision()) {
        const result = await provider.provision(job);
        if (result.success) return result;
      }
    }
    return { success: false, error: "No available compute provider" };
  }
}
