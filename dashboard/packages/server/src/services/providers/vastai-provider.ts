import type { FastifyBaseLogger } from "fastify";
import type { TrainingJob, GpuTierSpec } from "@tidal/shared";
import type {
  ComputeProvider,
  ProvisionResult,
} from "../compute-provider.js";
import type { GpuTier } from "../job-policy.js";
import { RateLimiter, type RateLimiterConfig } from "./rate-limiter.js";

const VASTAI_API = "https://console.vast.ai/api/v0";
const TERMINAL_INSTANCE_STATUSES = new Set(["exited", "offline", "error"]);
const MIN_GPU_RAM_MB = 48_000;
const MIN_CPU_CORES = 16;

const DEFAULT_DOCKER_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime";
const DEFAULT_GPU_TIERS: Record<string, GpuTierSpec> = {
  standard: { minGpuRamMb: MIN_GPU_RAM_MB, minCpuCores: MIN_CPU_CORES },
};

/**
 * Network and reliability constraints for Vast.ai offer search.
 * Progressive relaxation allows finding GPUs when strict constraints yield nothing.
 */
export interface ProvisionConstraints {
  inetDownMbps: number;
  inetUpMbps: number;
  reliability: number;
}

/** Default 3-tier relaxation schedule. */
const DEFAULT_RELAXATION_TIERS: ProvisionConstraints[] = [
  { inetDownMbps: 800, inetUpMbps: 800, reliability: 0.99 }, // Tier 0: strict
  { inetDownMbps: 400, inetUpMbps: 400, reliability: 0.97 }, // Tier 1: relaxed
  { inetDownMbps: 200, inetUpMbps: 200, reliability: 0.95 }, // Tier 2: loose
];

/** Default rate limiter: conservative vs Vast.ai's 4.5 req/sec threshold. */
const DEFAULT_RATE_LIMITER: RateLimiterConfig = {
  capacity: 4,
  refillRatePerMs: 1 / 1000, // 1 token/sec
};

/** Max retries on 429 responses before giving up. */
const MAX_429_RETRIES = 3;

export interface VastAIProviderConfig {
  apiKey: string | null;
  dashboardUrl: string | null;
  authToken: string | null;
  repoUrl: string | null;
  log: FastifyBaseLogger;
  dockerImage?: string;
  gpuTiers?: Record<string, GpuTierSpec>;
  relaxationTiers?: ProvisionConstraints[];
  rateLimiter?: RateLimiterConfig;
}

interface VastOffer {
  id: number;
  gpu_name: string;
  gpu_ram: number;
  dph_total: number;
  rentable: boolean;
  num_gpus: number;
  host_id?: number;
  machine_id?: number;
  gpu_mem_bw?: number;
  total_flops?: number;
  dlperf?: number;
  dlperf_per_dphtotal?: number;
  cpu_name?: string;
  cpu_cores?: number;
  cpu_cores_effective?: number;
  cpu_ram?: number;
  disk_name?: string;
  disk_bw?: number;
  disk_space?: number;
  inet_down?: number;
  inet_up?: number;
  mobo_name?: string;
  cuda_max_good?: number;
  reliability2?: number;
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
  private relaxationTiers: ProvisionConstraints[];
  private limiter: RateLimiter;

  constructor(config: VastAIProviderConfig) {
    this.apiKey = config.apiKey;
    this.dashboardUrl = config.dashboardUrl;
    this.authToken = config.authToken;
    this.repoUrl = config.repoUrl;
    this.log = config.log;
    this.dockerImage = config.dockerImage ?? DEFAULT_DOCKER_IMAGE;
    this.gpuTiers = config.gpuTiers ?? DEFAULT_GPU_TIERS;
    this.relaxationTiers = config.relaxationTiers ?? DEFAULT_RELAXATION_TIERS;
    this.limiter = new RateLimiter(config.rateLimiter ?? DEFAULT_RATE_LIMITER);
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
      const onStartScript = this.buildOnStartScript(job.jobId, job);

      // Try each constraint tier in order, relaxing on each failure
      for (let tierIdx = 0; tierIdx < this.relaxationTiers.length; tierIdx++) {
        const constraints = this.relaxationTiers[tierIdx];

        const offers = await this.findOffers(gpuTier ?? "standard", constraints);
        if (offers.length === 0) {
          this.log.info(
            { tierIdx, inetDown: constraints.inetDownMbps, reliability: constraints.reliability },
            "No vast.ai offers at this constraint tier, relaxing...",
          );
          continue;
        }

        let lastError: Error | undefined;
        let advanceToNextTier = false;

        for (const offer of offers) {
          this.log.info(
            { offerId: offer.id, gpu: offer.gpu_name, cost: offer.dph_total, tierIdx },
            "Trying vast.ai offer",
          );

          try {
            const instanceId = await this.createInstance(offer.id, onStartScript);
            return {
              success: true,
              meta: this.buildProviderMeta(offer, instanceId, tierIdx),
            };
          } catch (err) {
            lastError = err instanceof Error ? err : new Error(String(err));
            if (this.isRetryableOfferError(lastError)) {
              this.log.warn(
                { offerId: offer.id, err: lastError.message },
                "Offer unavailable, trying next",
              );
              continue;
            }
            // Non-retryable offer error — skip remaining offers in this tier
            this.log.warn(
              { offerId: offer.id, tierIdx, err: lastError.message },
              "Non-retryable offer error, moving to next constraint tier",
            );
            advanceToNextTier = true;
            break;
          }
        }

        if (!advanceToNextTier && lastError) {
          // All retryable offers exhausted — move to next tier
          this.log.info({ tierIdx }, "All offers in tier exhausted, relaxing constraints...");
        }
      }

      return {
        success: false,
        error: "No suitable vast.ai GPU offers found after all constraint tiers",
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
      const res = await this.fetch(`${VASTAI_API}/instances/${instanceId}/`, {
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
    const res = await this.fetch(`${VASTAI_API}/instances/${instanceId}/`, {
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

  /**
   * Rate-limited fetch wrapper with 429 retry.
   * Acquires a token from the rate limiter before each attempt,
   * and retries up to MAX_429_RETRIES times on 429 responses.
   */
  private async fetch(
    input: string | URL | Request,
    init?: RequestInit,
  ): Promise<Response> {
    for (let attempt = 0; attempt <= MAX_429_RETRIES; attempt++) {
      await this.limiter.acquire();
      const res = await globalThis.fetch(input, init);

      if (res.status !== 429) {
        return res;
      }

      if (attempt >= MAX_429_RETRIES) {
        throw new Error(
          `vast.ai API rate limited after ${MAX_429_RETRIES} retries (429 Too Many Requests)`,
        );
      }

      const retryAfter = res.headers.get("Retry-After") ?? undefined;
      const delay = this.limiter.backoff429(attempt, retryAfter);
      this.log.warn({ attempt, delay, retryAfter }, "vast.ai 429 — backing off before retry");
      await new Promise<void>((resolve) => setTimeout(resolve, delay));
    }

    // Unreachable but satisfies TypeScript
    throw new Error("vast.ai fetch: unexpected loop exit");
  }

  private async findOffers(tier: GpuTier, constraints: ProvisionConstraints): Promise<VastOffer[]> {
    const tierSpec = this.gpuTiers[tier] ?? DEFAULT_GPU_TIERS["standard"];
    const query = JSON.stringify({
      gpu_ram: { gte: tierSpec.minGpuRamMb },
      cpu_cores_effective: { gte: tierSpec.minCpuCores },
      inet_down: { gte: constraints.inetDownMbps },
      inet_up: { gte: constraints.inetUpMbps },
      rentable: { eq: true },
      num_gpus: { eq: 1 },
      reliability2: { gte: constraints.reliability },
      order: [["dph_total", "asc"]],
      type: "on-demand",
      limit: 10,
    });

    const res = await this.fetch(`${VASTAI_API}/bundles?q=${encodeURIComponent(query)}`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
    });

    if (!res.ok) {
      throw new Error(`vast.ai search failed: ${res.status} ${res.statusText}`);
    }

    const data = (await res.json()) as { offers?: VastOffer[] };
    return data.offers ?? [];
  }

  private async createInstance(offerId: number, onStartScript: string): Promise<number> {
    const res = await this.fetch(`${VASTAI_API}/asks/${offerId}/`, {
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

  private isRetryableOfferError(err: Error): boolean {
    return err.message.includes("no_such_ask");
  }

  private buildProviderMeta(
    offer: VastOffer,
    instanceId: number,
    constraintTier: number,
  ): Record<string, unknown> {
    return {
      instanceId,
      offerId: offer.id,
      hostId: offer.host_id ?? null,
      machineId: offer.machine_id ?? null,
      gpuName: offer.gpu_name,
      numGpus: offer.num_gpus,
      gpuRamMb: offer.gpu_ram ?? null,
      gpuMemBwGbps: offer.gpu_mem_bw ?? null,
      totalFlops: offer.total_flops ?? null,
      dlPerf: offer.dlperf ?? null,
      dlPerfPerDphTotal: offer.dlperf_per_dphtotal ?? null,
      cpuName: offer.cpu_name ?? null,
      cpuCores: offer.cpu_cores ?? null,
      cpuCoresEffective: offer.cpu_cores_effective ?? null,
      cpuRamMb: offer.cpu_ram ?? null,
      diskName: offer.disk_name ?? null,
      diskBwMbps: offer.disk_bw ?? null,
      diskSpaceGb: offer.disk_space ?? null,
      inetDownMbps: offer.inet_down ?? null,
      inetUpMbps: offer.inet_up ?? null,
      moboName: offer.mobo_name ?? null,
      cudaMaxGood: offer.cuda_max_good ?? null,
      reliability: offer.reliability2 ?? null,
      costPerHour: offer.dph_total,
      constraintTier,
      capturedAt: Date.now(),
    };
  }

  private buildOnStartScript(jobId: string, job: TrainingJob): string {
    const lines = [
      "#!/bin/bash",
      "set -e",
      "apt-get update && apt-get install -y git",
      `git clone ${this.repoUrl} /workspace/tidal`,
      "cd /workspace/tidal",
    ];

    // Clone user plugin repo if specified
    const { pluginRepoUrl, pluginName } = job.config;
    if (pluginRepoUrl && pluginName) {
      lines.push(`git clone ${pluginRepoUrl} plugins/${pluginName}`);
    }

    lines.push(
      "pip install -r requirements.txt",
      `python worker_agent.py --job-id ${jobId} --api-url ${this.dashboardUrl} --auth-token ${this.authToken}`,
    );

    return lines.join("\n");
  }
}
