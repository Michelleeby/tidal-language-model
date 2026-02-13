import type Redis from "ioredis";
import type { TrainingJob, JobSignal, JobStatus, RedisConfig } from "@tidal/shared";

const TERMINAL_STATUSES: Set<JobStatus> = new Set([
  "completed",
  "failed",
  "cancelled",
]);

export interface JobStoreConfig {
  /** TTL for signal keys in seconds (default 300). */
  signalTtl?: number;
  /** TTL for heartbeat keys in seconds (default 30). */
  heartbeatTtl?: number;
}

export interface JobStoreRedisKeys {
  jobsHash: string;
  activeSet: string;
  signalPrefix: string;
  heartbeatPrefix: string;
  updatesChannel: string;
}

const DEFAULT_REDIS_KEYS: JobStoreRedisKeys = {
  jobsHash: "tidal:jobs",
  activeSet: "tidal:jobs:active",
  signalPrefix: "tidal:job:",
  heartbeatPrefix: "tidal:worker:",
  updatesChannel: "tidal:job:updates",
};

/**
 * Build JobStoreRedisKeys from a plugin's RedisConfig.
 */
export function jobStoreKeysFromManifest(redis: RedisConfig): JobStoreRedisKeys {
  return {
    jobsHash: redis.jobsHash,
    activeSet: redis.jobsActiveSet,
    signalPrefix: redis.signalPrefix,
    heartbeatPrefix: redis.heartbeatPrefix,
    updatesChannel: redis.updatesChannel,
  };
}

export class JobStore {
  private signalTtl: number;
  private heartbeatTtl: number;
  private keys: JobStoreRedisKeys;

  constructor(
    private redis: Redis | null,
    config?: JobStoreConfig,
    redisKeys?: JobStoreRedisKeys,
  ) {
    this.signalTtl = config?.signalTtl ?? 300;
    this.heartbeatTtl = config?.heartbeatTtl ?? 30;
    this.keys = redisKeys ?? DEFAULT_REDIS_KEYS;
  }

  private ensureRedis(): Redis {
    if (!this.redis) throw new Error("Redis unavailable");
    return this.redis;
  }

  async create(job: TrainingJob): Promise<void> {
    const r = this.ensureRedis();
    await r.hset(this.keys.jobsHash, job.jobId, JSON.stringify(job));
    if (!TERMINAL_STATUSES.has(job.status)) {
      await r.sadd(this.keys.activeSet, job.jobId);
    }
  }

  async get(jobId: string): Promise<TrainingJob | null> {
    const r = this.ensureRedis();
    const raw = await r.hget(this.keys.jobsHash, jobId);
    return raw ? (JSON.parse(raw) as TrainingJob) : null;
  }

  async update(
    jobId: string,
    patch: Partial<TrainingJob>,
  ): Promise<TrainingJob | null> {
    const r = this.ensureRedis();
    const raw = await r.hget(this.keys.jobsHash, jobId);
    if (!raw) return null;

    const job: TrainingJob = { ...JSON.parse(raw), ...patch, updatedAt: Date.now() };
    await r.hset(this.keys.jobsHash, jobId, JSON.stringify(job));

    if (TERMINAL_STATUSES.has(job.status)) {
      await r.srem(this.keys.activeSet, jobId);
    }

    await r.publish(this.keys.updatesChannel, JSON.stringify({ jobId }));
    return job;
  }

  async list(): Promise<TrainingJob[]> {
    const r = this.ensureRedis();
    const all = await r.hgetall(this.keys.jobsHash);
    return Object.values(all).map((raw) => JSON.parse(raw) as TrainingJob);
  }

  async listActive(): Promise<TrainingJob[]> {
    const r = this.ensureRedis();
    const ids = await r.smembers(this.keys.activeSet);
    if (ids.length === 0) return [];
    const raws = await r.hmget(this.keys.jobsHash, ...ids);
    return raws.filter(Boolean).map((raw) => JSON.parse(raw!) as TrainingJob);
  }

  async sendSignal(jobId: string, signal: JobSignal): Promise<void> {
    const r = this.ensureRedis();
    const key = `${this.keys.signalPrefix}${jobId}:signal`;
    await r.set(key, signal, "EX", this.signalTtl);
    await r.publish(this.keys.updatesChannel, JSON.stringify({ jobId, signal }));
  }

  async readSignal(jobId: string): Promise<JobSignal | null> {
    const r = this.ensureRedis();
    const val = await r.get(`${this.keys.signalPrefix}${jobId}:signal`);
    return val as JobSignal | null;
  }

  async clearSignal(jobId: string): Promise<void> {
    const r = this.ensureRedis();
    await r.del(`${this.keys.signalPrefix}${jobId}:signal`);
  }

  async getHeartbeat(jobId: string): Promise<number | null> {
    const r = this.ensureRedis();
    const val = await r.get(`${this.keys.heartbeatPrefix}${jobId}:heartbeat`);
    return val ? parseFloat(val) : null;
  }

  async setHeartbeat(jobId: string): Promise<void> {
    const r = this.ensureRedis();
    const key = `${this.keys.heartbeatPrefix}${jobId}:heartbeat`;
    await r.set(key, String(Date.now() / 1000), "EX", this.heartbeatTtl);
  }
}
