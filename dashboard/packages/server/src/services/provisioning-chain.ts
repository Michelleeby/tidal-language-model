import type { ComputeProviderType } from "@tidal/shared";
import type { ComputeProvider } from "./compute-provider.js";

export class ProvisioningChain {
  private byType: Map<ComputeProviderType, ComputeProvider>;

  constructor(providers: ComputeProvider[]) {
    this.byType = new Map(providers.map((p) => [p.type, p]));
  }

  getProvider(type: ComputeProviderType): ComputeProvider | undefined {
    return this.byType.get(type);
  }
}
