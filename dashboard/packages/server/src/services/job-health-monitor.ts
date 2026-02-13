import type { FastifyBaseLogger } from "fastify";
import type { JobStatus } from "@tidal/shared";
import type { JobStore } from "./job-store.js";
import type { ProvisioningChain } from "./provisioning-chain.js";
import type { WorkerSpawner } from "./worker-spawner.js";

export interface HealthMonitorConfig {
  healthCheckIntervalMs: number;
  heartbeatTimeoutMs: number;
  staleStartupTimeoutMs: number;
  remoteStartupTimeoutMs: number;
}

const TERMINAL_STATUSES: Set<JobStatus> = new Set([
  "completed",
  "failed",
  "cancelled",
]);

export class JobHealthMonitor {
  private timer: ReturnType<typeof setInterval> | null = null;

  constructor(
    private store: JobStore,
    private chain: ProvisioningChain,
    private spawner: WorkerSpawner,
    private config: HealthMonitorConfig,
    private onJobFailed: (
      jobId: string,
      status: "completed" | "failed",
      error?: string,
    ) => Promise<void>,
    private log: FastifyBaseLogger,
  ) {}

  start(): void {
    this.timer = setInterval(
      () => this.healthCheck(),
      this.config.healthCheckIntervalMs,
    );
  }

  stop(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  private async healthCheck(): Promise<void> {
    try {
      const active = await this.store.listActive();
      for (const job of active) {
        if (TERMINAL_STATUSES.has(job.status)) continue;

        const provider = this.chain.getProvider(job.provider);
        const isRemote = provider?.isRemote ?? false;

        // Jobs stuck in startup phases (pending/provisioning/starting)
        if (
          job.status === "pending" ||
          job.status === "provisioning" ||
          job.status === "starting"
        ) {
          const age = Date.now() - job.updatedAt;
          const timeout = isRemote
            ? this.config.remoteStartupTimeoutMs
            : this.config.staleStartupTimeoutMs;
          if (age > timeout) {
            this.log.warn(
              { jobId: job.jobId, status: job.status, ageMs: age },
              "Job stuck in startup — marking failed",
            );
            await this.onJobFailed(
              job.jobId,
              "failed",
              `Stuck in ${job.status} for ${Math.round(age / 1000)}s`,
            );
            continue;
          }

          // Remote jobs with a provisioned instance: check if still alive
          // (60s grace period to avoid checking right after provisioning)
          if (isRemote && provider && job.providerMeta && age > 60_000) {
            try {
              const alive = await provider.isAlive(job);
              if (!alive) {
                this.log.warn(
                  { jobId: job.jobId, status: job.status },
                  "Remote instance no longer alive during startup — marking failed",
                );
                await this.onJobFailed(
                  job.jobId,
                  "failed",
                  "Remote instance terminated during startup",
                );
              }
            } catch {
              // isAlive check failed — don't fail the job, let the timeout handle it
            }
          }
          continue;
        }

        // Running/completing/stopping jobs — check heartbeat
        const heartbeat = await this.store.getHeartbeat(job.jobId);
        if (heartbeat === null) {
          // Remote jobs: check if the instance is still alive
          if (isRemote) {
            if (provider) {
              try {
                const alive = await provider.isAlive(job);
                if (!alive) {
                  this.log.warn(
                    { jobId: job.jobId },
                    "Remote instance gone, no heartbeat — marking failed",
                  );
                  await this.onJobFailed(
                    job.jobId,
                    "failed",
                    "Remote instance terminated",
                  );
                }
              } catch {
                // isAlive check failed — skip for now
              }
            }
            continue;
          }
          // Local jobs: check if worker process is still running
          if (!this.spawner.isRunning(job.jobId)) {
            this.log.warn(
              { jobId: job.jobId },
              "Worker not running, no heartbeat — marking failed",
            );
            await this.onJobFailed(
              job.jobId,
              "failed",
              "Worker process not found",
            );
          }
          continue;
        }

        const age = Date.now() - heartbeat * 1000;
        if (age > this.config.heartbeatTimeoutMs) {
          // Remote jobs: check if the instance is still alive before killing
          if (isRemote && provider) {
            try {
              const alive = await provider.isAlive(job);
              if (alive) {
                this.log.info(
                  { jobId: job.jobId, ageMs: age },
                  "Heartbeat stale but instance still alive — skipping",
                );
                continue;
              }
            } catch {
              // isAlive check failed, proceed with timeout
            }
          }
          this.log.warn(
            { jobId: job.jobId, ageMs: age },
            "Heartbeat stale — marking failed",
          );
          await this.onJobFailed(
            job.jobId,
            "failed",
            "Worker heartbeat timeout",
          );
        }
      }
    } catch {
      // Non-fatal health check error
    }
  }
}
