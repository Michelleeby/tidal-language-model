import { nanoid } from "nanoid";
import type { FastifyBaseLogger } from "fastify";
import type {
  ComputeProviderType,
  CreateJobRequest,
  JobSignal,
  JobStatus,
  JobType,
  TrainingJob,
} from "@tidal/shared";
import type { JobStore } from "./job-store.js";
import type { ProvisioningChain } from "./provisioning-chain.js";
import type { WorkerSpawner } from "./worker-spawner.js";
import type { SSEManager } from "./sse-manager.js";
import type { ExperimentArchiver } from "./experiment-archiver.js";

export interface OrchestratorConfig {
  defaultProvider: ComputeProviderType;
  healthCheckIntervalMs: number;
  heartbeatTimeoutMs: number;
  /** How long a job can sit in pending/provisioning/starting before being marked failed */
  staleStartupTimeoutMs: number;
  /** How long a remote job can sit in starting before being marked failed (default 5 min) */
  remoteStartupTimeoutMs: number;
}

const DEFAULT_CONFIG: OrchestratorConfig = {
  defaultProvider: "local",
  healthCheckIntervalMs: 15_000,
  heartbeatTimeoutMs: 180_000,
  staleStartupTimeoutMs: 30_000,
  remoteStartupTimeoutMs: 900_000,
};

const TERMINAL_STATUSES: Set<JobStatus> = new Set([
  "completed",
  "failed",
  "cancelled",
]);

export class JobOrchestrator {
  private healthTimer: ReturnType<typeof setInterval> | null = null;
  private config: OrchestratorConfig;

  constructor(
    private store: JobStore,
    private chain: ProvisioningChain,
    private spawner: WorkerSpawner,
    private sseManager: SSEManager,
    private log: FastifyBaseLogger,
    config?: Partial<OrchestratorConfig>,
    private archiver?: ExperimentArchiver,
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.healthTimer = setInterval(
      () => this.healthCheck(),
      this.config.healthCheckIntervalMs,
    );
  }

  async createJob(request: CreateJobRequest): Promise<TrainingJob> {
    // Enforce one active LM job at a time
    if (request.type === "lm-training") {
      const active = await this.getActiveJob("lm-training");
      if (active) {
        throw new Error("An LM training job is already running");
      }
    }

    const now = Date.now();
    const job: TrainingJob = {
      jobId: nanoid(12),
      type: request.type,
      status: "pending",
      provider: request.provider ?? this.config.defaultProvider,
      config: {
        type: request.type,
        configPath: request.configPath,
        resumeExpDir: request.resumeExpDir,
        checkpoint: request.checkpoint,
        rlConfigPath: request.rlConfigPath,
        timesteps: request.timesteps,
      },
      createdAt: now,
      updatedAt: now,
    };

    await this.store.create(job);
    this.log.info({ jobId: job.jobId, type: job.type }, "Job created");

    try {
      // Provision
      const provisioning = await this.store.update(job.jobId, {
        status: "provisioning",
      });
      this.broadcast(provisioning!);

      const provider =
        this.chain.getProvider(job.provider) ??
        this.chain.getProvider(this.config.defaultProvider);

      if (!provider) {
        const failed = await this.store.update(job.jobId, {
          status: "failed",
          error: `No provider available for type: ${job.provider}`,
          completedAt: Date.now(),
        });
        this.broadcast(failed!);
        return failed!;
      }

      const result = await provider.provision(job);
      if (!result.success) {
        const failed = await this.store.update(job.jobId, {
          status: "failed",
          error: result.error ?? "Provisioning failed",
          providerMeta: result.meta,
          completedAt: Date.now(),
        });
        this.broadcast(failed!);
        return failed!;
      }

      // Spawn worker
      const starting = await this.store.update(job.jobId, {
        status: "starting",
        providerMeta: result.meta,
      });
      this.broadcast(starting!);

      if (provider.isRemote) {
        // Remote providers: worker connects via API and sets itself to "running"
        this.log.info({ jobId: job.jobId }, "Remote job — waiting for worker to connect");
        return starting!;
      }

      this.spawner.spawnLocal(job.jobId, job.config);

      const running = await this.store.update(job.jobId, {
        status: "running",
        startedAt: Date.now(),
      });
      this.broadcast(running!);

      return running!;
    } catch (err) {
      // If anything goes wrong during provision/spawn, mark the job as failed
      this.log.error({ jobId: job.jobId, err }, "Job creation failed");
      const failed = await this.store.update(job.jobId, {
        status: "failed",
        error: err instanceof Error ? err.message : String(err),
        completedAt: Date.now(),
      });
      this.broadcast(failed!);
      return failed!;
    }
  }

  async signalJob(jobId: string, signal: JobSignal): Promise<TrainingJob> {
    const job = await this.store.get(jobId);
    if (!job) throw new Error(`Job not found: ${jobId}`);

    if (job.status !== "running" && job.status !== "completing") {
      throw new Error(`Cannot signal job in status: ${job.status}`);
    }

    await this.store.sendSignal(jobId, signal);

    const newStatus = signal === "complete" ? "completing" : "stopping";
    const updated = await this.store.update(jobId, { status: newStatus });
    this.broadcast(updated!);

    if (signal === "stop" && this.spawner.isRunning(jobId)) {
      this.spawner.kill(jobId);
    }

    this.log.info({ jobId, signal }, "Signal sent to job");
    return updated!;
  }

  async cancelJob(jobId: string): Promise<TrainingJob> {
    const job = await this.store.get(jobId);
    if (!job) throw new Error(`Job not found: ${jobId}`);

    if (TERMINAL_STATUSES.has(job.status)) {
      throw new Error(`Job already in terminal status: ${job.status}`);
    }

    // Kill local worker if it exists
    if (this.spawner.isRunning(jobId)) {
      this.spawner.kill(jobId);
    }

    // Deprovision remote instances
    const provider = this.chain.getProvider(job.provider);
    if (provider?.isRemote) {
      await provider.deprovision(job);
    }

    const updated = await this.store.update(jobId, {
      status: "cancelled",
      completedAt: Date.now(),
    });
    this.broadcast(updated!);

    this.log.info({ jobId }, "Job cancelled");
    return updated!;
  }

  async getActiveJob(type?: JobType): Promise<TrainingJob | null> {
    const active = await this.store.listActive();
    if (type) {
      return active.find((j) => j.type === type) ?? null;
    }
    return active[0] ?? null;
  }

  async handleJobComplete(
    jobId: string,
    status: "completed" | "failed",
    error?: string,
  ): Promise<void> {
    const job = await this.store.get(jobId);

    const patch: Partial<TrainingJob> = {
      status,
      completedAt: Date.now(),
    };
    if (error) patch.error = error;

    const updated = await this.store.update(jobId, patch);
    if (updated) this.broadcast(updated);

    // Archive experiment data from Redis to disk before TTLs expire
    if (job?.experimentId && this.archiver) {
      this.archiver.archive(job.experimentId).catch((err) => {
        this.log.error({ expId: job.experimentId, err }, "Archival failed in handleJobComplete");
      });
    }

    // Deprovision remote instances on completion
    if (job) {
      const provider = this.chain.getProvider(job.provider);
      if (provider?.isRemote) {
        await provider.deprovision(job);
      }
    }
  }

  stop(): void {
    if (this.healthTimer) {
      clearInterval(this.healthTimer);
      this.healthTimer = null;
    }
    this.spawner.cleanup();
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
            await this.handleJobComplete(
              job.jobId,
              "failed",
              `Stuck in ${job.status} for ${Math.round(age / 1000)}s`,
            );
          }
          continue;
        }

        // Running/completing/stopping jobs — check heartbeat
        const heartbeat = await this.store.getHeartbeat(job.jobId);
        if (heartbeat === null) {
          // Remote jobs: rely on heartbeat only (no local process to check)
          if (isRemote) continue;
          // Local jobs: check if worker process is still running
          if (!this.spawner.isRunning(job.jobId)) {
            this.log.warn(
              { jobId: job.jobId },
              "Worker not running, no heartbeat — marking failed",
            );
            await this.handleJobComplete(
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
          await this.handleJobComplete(
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

  private broadcast(job: TrainingJob): void {
    this.sseManager.broadcastJobUpdate(job);
  }
}
