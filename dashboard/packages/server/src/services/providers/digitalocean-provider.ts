import type { FastifyBaseLogger } from "fastify";
import type { TrainingJob, GpuTierSpec } from "@tidal/shared";
import type {
  ComputeProvider,
  ProvisionResult,
} from "../compute-provider.js";
import type { GpuTier } from "../job-policy.js";

const DO_API = "https://api.digitalocean.com";
const TERMINAL_STATUSES = new Set(["off", "archive"]);

const DEFAULT_GPU_TIERS: Record<string, GpuTierSpec> = {
  standard: { minGpuRamMb: 48_000, minCpuCores: 16 },
};

export interface DigitalOceanProviderConfig {
  apiKey: string | null;
  dashboardUrl: string | null;
  authToken: string | null;
  repoUrl: string | null;
  log: FastifyBaseLogger;
  region: string;
  sshKey?: string | null;
  gpuTiers?: Record<string, GpuTierSpec>;
}

interface DOSize {
  slug: string;
  memory: number;
  vcpus: number;
  disk: number;
  price_monthly: number;
  price_hourly: number;
  regions: string[];
  available: boolean;
  description: string;
  gpu_info?: {
    count: number;
    vram: { amount: number; unit: string };
    model: string;
  };
}

export class DigitalOceanProvider implements ComputeProvider {
  readonly type = "digitalocean" as const;
  readonly isRemote = true;

  private apiKey: string | null;
  private dashboardUrl: string | null;
  private authToken: string | null;
  private repoUrl: string | null;
  private log: FastifyBaseLogger;
  private region: string;
  private sshKey: string | null;
  private gpuTiers: Record<string, GpuTierSpec>;

  constructor(config: DigitalOceanProviderConfig) {
    this.apiKey = config.apiKey;
    this.dashboardUrl = config.dashboardUrl;
    this.authToken = config.authToken;
    this.repoUrl = config.repoUrl;
    this.log = config.log;
    this.region = config.region;
    this.sshKey = config.sshKey ?? null;
    this.gpuTiers = config.gpuTiers ?? DEFAULT_GPU_TIERS;
  }

  async canProvision(): Promise<boolean> {
    return this.apiKey !== null && this.apiKey.length > 0;
  }

  async provision(job: TrainingJob, gpuTier?: GpuTier): Promise<ProvisionResult> {
    if (!this.apiKey) {
      return { success: false, error: "DigitalOcean API key not configured" };
    }
    if (!this.dashboardUrl || !this.authToken || !this.repoUrl) {
      return {
        success: false,
        error: "Missing dashboardUrl, authToken, or repoUrl for DigitalOcean remote worker",
      };
    }

    try {
      const sizes = await this.findGpuSizes(gpuTier ?? "standard");
      if (sizes.length === 0) {
        return { success: false, error: "No suitable DigitalOcean GPU sizes found" };
      }

      const bestSize = sizes[0];
      this.log.info(
        { slug: bestSize.slug, cost: bestSize.price_hourly },
        "Selected DigitalOcean GPU size",
      );

      const dropletId = await this.createDroplet(bestSize.slug, job.jobId);
      return {
        success: true,
        meta: {
          dropletId,
          sizeSlug: bestSize.slug,
          region: this.region,
          pricePerHour: bestSize.price_hourly,
        },
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.log.error({ err, jobId: job.jobId }, "DigitalOcean provision failed");
      return { success: false, error: `DigitalOcean provision failed: ${message}` };
    }
  }

  async deprovision(job: TrainingJob): Promise<void> {
    const dropletId = job.providerMeta?.dropletId;
    if (!dropletId || !this.apiKey) return;

    try {
      const res = await fetch(`${DO_API}/v2/droplets/${dropletId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      if (!res.ok) {
        this.log.warn(
          { dropletId, status: res.status },
          "DigitalOcean deprovision returned non-OK",
        );
      }
    } catch (err) {
      this.log.error({ err, dropletId }, "DigitalOcean deprovision failed");
    }
  }

  async isAlive(job: TrainingJob): Promise<boolean> {
    const dropletId = job.providerMeta?.dropletId;
    if (!dropletId || !this.apiKey) return false;

    const res = await fetch(`${DO_API}/v2/droplets/${dropletId}`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
    });

    if (res.status === 404) return false;

    if (!res.ok) {
      throw new Error(`DigitalOcean droplet status check failed: ${res.status}`);
    }

    const data = (await res.json()) as { droplet?: { status?: string } };
    if (!data.droplet?.status) {
      throw new Error("DigitalOcean returned no droplet status");
    }
    return !TERMINAL_STATUSES.has(data.droplet.status);
  }

  private async findGpuSizes(tier: GpuTier): Promise<DOSize[]> {
    const tierSpec = this.gpuTiers[tier] ?? DEFAULT_GPU_TIERS["standard"];

    const res = await fetch(`${DO_API}/v2/sizes?per_page=200`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
    });

    if (!res.ok) {
      throw new Error(`DigitalOcean sizes request failed: ${res.status} ${res.statusText}`);
    }

    const data = (await res.json()) as { sizes?: DOSize[] };
    const sizes = data.sizes ?? [];

    return sizes
      .filter((s) => {
        if (!s.available) return false;
        if (!s.regions.includes(this.region)) return false;
        if (!s.gpu_info) return false;
        // DO reports VRAM in GiB — convert to MiB for comparison
        const vramMb = s.gpu_info.vram.amount * 1024;
        if (vramMb < tierSpec.minGpuRamMb) return false;
        if (s.vcpus < tierSpec.minCpuCores) return false;
        return true;
      })
      .sort((a, b) => a.price_hourly - b.price_hourly);
  }

  private async createDroplet(sizeSlug: string, jobId: string): Promise<number> {
    const userData = this.buildUserData(jobId);

    const res = await fetch(`${DO_API}/v2/droplets`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name: `tidal-worker-${jobId}`,
        region: this.region,
        size: sizeSlug,
        image: "gpu-h100x1-base",
        user_data: userData,
        tags: ["tidal", "worker"],
        ...(this.sshKey ? { ssh_keys: [this.sshKey] } : {}),
      }),
    });

    if (!res.ok) {
      const body = await res.text();
      throw new Error(`DigitalOcean create droplet failed: ${res.status} ${body}`);
    }

    const data = (await res.json()) as { droplet?: { id?: number } };
    if (!data.droplet?.id) {
      throw new Error("DigitalOcean did not return a droplet ID");
    }

    return data.droplet.id;
  }

  private buildUserData(jobId: string): string {
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
