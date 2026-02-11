import type Redis from "ioredis";
import type { TrainingJob, JobSignal, JobStatus } from "@tidal/shared";

const JOBS_HASH = "tidal:jobs";
const ACTIVE_SET = "tidal:jobs:active";
const SIGNAL_PREFIX = "tidal:job:";
const UPDATES_CHANNEL = "tidal:job:updates";
const HEARTBEAT_PREFIX = "tidal:worker:";

const SIGNAL_TTL = 300;

const TERMINAL_STATUSES: Set<JobStatus> = new Set([
  "completed",
  "failed",
  "cancelled",
]);

export class JobStore {
  constructor(private redis: Redis | null) {}

  private ensureRedis(): Redis {
    if (!this.redis) throw new Error("Redis unavailable");
    return this.redis;
  }

  async create(job: TrainingJob): Promise<void> {
    const r = this.ensureRedis();
    await r.hset(JOBS_HASH, job.jobId, JSON.stringify(job));
    if (!TERMINAL_STATUSES.has(job.status)) {
      await r.sadd(ACTIVE_SET, job.jobId);
    }
  }

  async get(jobId: string): Promise<TrainingJob | null> {
    const r = this.ensureRedis();
    const raw = await r.hget(JOBS_HASH, jobId);
    return raw ? (JSON.parse(raw) as TrainingJob) : null;
  }

  async update(
    jobId: string,
    patch: Partial<TrainingJob>,
  ): Promise<TrainingJob | null> {
    const r = this.ensureRedis();
    const raw = await r.hget(JOBS_HASH, jobId);
    if (!raw) return null;

    const job: TrainingJob = { ...JSON.parse(raw), ...patch, updatedAt: Date.now() };
    await r.hset(JOBS_HASH, jobId, JSON.stringify(job));

    if (TERMINAL_STATUSES.has(job.status)) {
      await r.srem(ACTIVE_SET, jobId);
    }

    await r.publish(UPDATES_CHANNEL, JSON.stringify({ jobId }));
    return job;
  }

  async list(): Promise<TrainingJob[]> {
    const r = this.ensureRedis();
    const all = await r.hgetall(JOBS_HASH);
    return Object.values(all).map((raw) => JSON.parse(raw) as TrainingJob);
  }

  async listActive(): Promise<TrainingJob[]> {
    const r = this.ensureRedis();
    const ids = await r.smembers(ACTIVE_SET);
    if (ids.length === 0) return [];
    const raws = await r.hmget(JOBS_HASH, ...ids);
    return raws.filter(Boolean).map((raw) => JSON.parse(raw!) as TrainingJob);
  }

  async sendSignal(jobId: string, signal: JobSignal): Promise<void> {
    const r = this.ensureRedis();
    const key = `${SIGNAL_PREFIX}${jobId}:signal`;
    await r.set(key, signal, "EX", SIGNAL_TTL);
    await r.publish(UPDATES_CHANNEL, JSON.stringify({ jobId, signal }));
  }

  async readSignal(jobId: string): Promise<JobSignal | null> {
    const r = this.ensureRedis();
    const val = await r.get(`${SIGNAL_PREFIX}${jobId}:signal`);
    return val as JobSignal | null;
  }

  async clearSignal(jobId: string): Promise<void> {
    const r = this.ensureRedis();
    await r.del(`${SIGNAL_PREFIX}${jobId}:signal`);
  }

  async getHeartbeat(jobId: string): Promise<number | null> {
    const r = this.ensureRedis();
    const val = await r.get(`${HEARTBEAT_PREFIX}${jobId}:heartbeat`);
    return val ? parseFloat(val) : null;
  }
}
