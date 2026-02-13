import type { FastifyBaseLogger } from "fastify";
import type { TrainingJob, GpuTierSpec } from "@tidal/shared";
import type {
  ComputeProvider,
  ProvisionResult,
} from "../compute-provider.js";
import type { GpuTier } from "../job-policy.js";

const VASTAI_API = "https://console.vast.ai/api/v0";
const TERMINAL_INSTANCE_STATUSES = new Set(["exited", "offline", "error"]);
const MIN_INET_DOWN_MBPS = 800;
const MIN_INET_UP_MBPS = 800;
const MIN_RELIABILITY = 0.99;

const DEFAULT_DOCKER_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime";
const DEFAULT_GPU_TIERS: Record<string, GpuTierSpec> = {
  standard: { minGpuRamMb: 16_000, minCpuCores: 16 },
};

export interface VastAIProviderConfig {
  apiKey: string | null;
  dashboardUrl: string | null;
  authToken: string | null;
  repoUrl: string | null;
  log: FastifyBaseLogger;
  dockerImage?: string;
  gpuTiers?: Record<string, GpuTierSpec>;
}

interface VastOffer {
  id: number;
  gpu_name: string;
  gpu_ram: number;
  dph_total: number;
  rentable: boolean;
  num_gpus: number;
}

export class VastAIProvider implements ComputeProvider {
  readonly type = "vastai" as const;
  readonly isRemote = true;

  private apiKey: string | null;
  private dashboardUrl: string | null;
  private authToken: string | null;
  private repoUrl: string | null;
  private log: FastifyBaseLogger;
  private dockerImage: string;
  private gpuTiers: Record<string, GpuTierSpec>;

  constructor(config: VastAIProviderConfig) {
    this.apiKey = config.apiKey;
    this.dashboardUrl = config.dashboardUrl;
    this.authToken = config.authToken;
    this.repoUrl = config.repoUrl;
    this.log = config.log;
    this.dockerImage = config.dockerImage ?? DEFAULT_DOCKER_IMAGE;
    this.gpuTiers = config.gpuTiers ?? DEFAULT_GPU_TIERS;
  }

  async canProvision(): Promise<boolean> {
    return this.apiKey !== null && this.apiKey.length > 0;
  }

  async provision(job: TrainingJob, gpuTier?: GpuTier): Promise<ProvisionResult> {
    if (!this.apiKey) {
      return { success: false, error: "vast.ai API key not configured" };
    }
    if (!this.dashboardUrl || !this.authToken || !this.repoUrl) {
      return {
        success: false,
        error: "Missing dashboardUrl, authToken, or repoUrl for vast.ai remote worker",
      };
    }

    try {
      // 1. Search for cheapest GPU offer
      const offer = await this.findCheapestOffer(gpuTier ?? "standard");
      if (!offer) {
        return { success: false, error: "No suitable vast.ai GPU offers found" };
      }

      this.log.info(
        { offerId: offer.id, gpu: offer.gpu_name, cost: offer.dph_total },
        "Selected vast.ai offer",
      );

      // 2. Create instance with on-start script
      const onStartScript = this.buildOnStartScript(job.jobId);
      const instanceId = await this.createInstance(offer.id, onStartScript);

      return {
        success: true,
        meta: {
          instanceId,
          offerId: offer.id,
          gpuName: offer.gpu_name,
          costPerHour: offer.dph_total,
        },
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.log.error({ err, jobId: job.jobId }, "vast.ai provision failed");
      return { success: false, error: `vast.ai provision failed: ${message}` };
    }
  }

  async deprovision(job: TrainingJob): Promise<void> {
    const instanceId = job.providerMeta?.instanceId;
    if (!instanceId || !this.apiKey) return;

    try {
      const res = await fetch(`${VASTAI_API}/instances/${instanceId}/`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      if (!res.ok) {
        const body = await res.text().catch(() => "");
        this.log.warn(
          { instanceId, status: res.status, body },
          "vast.ai deprovision returned non-OK",
        );
      }
    } catch (err) {
      this.log.error({ err, instanceId }, "vast.ai deprovision failed");
    }
  }

  async isAlive(job: TrainingJob): Promise<boolean> {
    const instanceId = job.providerMeta?.instanceId;
    if (!instanceId || !this.apiKey) return false;

    // No try/catch -- let network errors propagate as throws so callers
    // can distinguish "confirmed dead" (false) from "inconclusive" (throw).
    const res = await fetch(`${VASTAI_API}/instances/${instanceId}/`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
    });

    if (res.status === 404) return false; // instance destroyed

    if (!res.ok) {
      throw new Error(`vast.ai instance status check failed: ${res.status}`);
    }

    const data = (await res.json()) as { actual_status?: string };
    if (!data.actual_status) {
      throw new Error("vast.ai returned no actual_status");
    }
    return !TERMINAL_INSTANCE_STATUSES.has(data.actual_status);
  }

  private async findCheapestOffer(tier: GpuTier): Promise<VastOffer | null> {
    const tierSpec = this.gpuTiers[tier] ?? DEFAULT_GPU_TIERS["standard"];
    const query = JSON.stringify({
      gpu_ram: { gte: tierSpec.minGpuRamMb },
      cpu_cores_effective: { gte: tierSpec.minCpuCores },
      inet_down: { gte: MIN_INET_DOWN_MBPS },
      inet_up: { gte: MIN_INET_UP_MBPS },
      rentable: { eq: true },
      num_gpus: { eq: 1 },
      reliability2: { gte: MIN_RELIABILITY },
      order: [["dph_total", "asc"]],
      type: "on-demand",
      limit: 10,
    });

    const res = await fetch(`${VASTAI_API}/bundles?q=${encodeURIComponent(query)}`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
    });

    if (!res.ok) {
      throw new Error(`vast.ai search failed: ${res.status} ${res.statusText}`);
    }

    const data = (await res.json()) as { offers?: VastOffer[] };
    const offers = data.offers ?? [];
    return offers[0] ?? null;
  }

  private async createInstance(offerId: number, onStartScript: string): Promise<number> {
    const res = await fetch(`${VASTAI_API}/asks/${offerId}/`, {
      method: "PUT",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        client_id: "me",
        image: this.dockerImage,
        disk: 20,
        onstart: onStartScript,
      }),
    });

    if (!res.ok) {
      const body = await res.text();
      throw new Error(`vast.ai create instance failed: ${res.status} ${body}`);
    }

    const data = (await res.json()) as { new_contract?: number };
    if (!data.new_contract) {
      throw new Error("vast.ai did not return an instance ID");
    }

    return data.new_contract;
  }

  private buildOnStartScript(jobId: string): string {
    return [
      "#!/bin/bash",
      "set -e",
      "apt-get update && apt-get install -y git",
      `git clone ${this.repoUrl} /workspace/tidal`,
      "cd /workspace/tidal",
      "pip install -r requirements.txt",
      `python worker_agent.py --job-id ${jobId} --api-url ${this.dashboardUrl} --auth-token ${this.authToken}`,
    ].join("\n");
  }
}
