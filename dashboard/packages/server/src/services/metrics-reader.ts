import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import readline from "node:readline";
import type Redis from "ioredis";
import type { MetricPoint, TrainingStatus, RLTrainingMetrics, MetricsConfig } from "@tidal/shared";

export interface MetricsReaderConfig {
  /** Redis key prefix, e.g. "tidal" */
  redisPrefix: string;
  /** LM metrics directory name, e.g. "dashboard_metrics" */
  lmDirectory: string;
  /** LM history filename, e.g. "metrics.jsonl" */
  lmHistoryFile: string;
  /** LM status filename, e.g. "status.json" */
  lmStatusFile: string;
  /** LM latest filename, e.g. "latest.json" */
  lmLatestFile: string;
  /** RL metrics directory name, e.g. "rl_metrics" */
  rlDirectory: string;
  /** RL metrics filename, e.g. "rl_training_metrics.json" */
  rlMetricsFile: string;
}

/**
 * Build MetricsReaderConfig from a plugin's MetricsConfig.
 */
export function metricsReaderConfigFromManifest(metrics: MetricsConfig): MetricsReaderConfig {
  return {
    redisPrefix: metrics.redisPrefix,
    lmDirectory: metrics.lm.directory,
    lmHistoryFile: metrics.lm.historyFile,
    lmStatusFile: metrics.lm.statusFile,
    lmLatestFile: metrics.lm.latestFile,
    rlDirectory: metrics.rl.directory,
    rlMetricsFile: metrics.rl.metricsFile,
  };
}

/** Default config matching the original hardcoded values. */
export const DEFAULT_METRICS_CONFIG: MetricsReaderConfig = {
  redisPrefix: "tidal",
  lmDirectory: "dashboard_metrics",
  lmHistoryFile: "metrics.jsonl",
  lmStatusFile: "status.json",
  lmLatestFile: "latest.json",
  rlDirectory: "rl_metrics",
  rlMetricsFile: "rl_training_metrics.json",
};

/**
 * Reads metrics from Redis (primary) or JSONL files (fallback).
 */
export class MetricsReader {
  private mc: MetricsReaderConfig;

  constructor(
    private redis: Redis | null,
    private experimentsDir: string,
    metricsConfig?: MetricsReaderConfig,
  ) {
    this.mc = metricsConfig ?? DEFAULT_METRICS_CONFIG;
  }

  /** Read recent metrics (last `window` points). */
  async getRecentMetrics(
    expId: string,
    window: number,
  ): Promise<MetricPoint[]> {
    // Try Redis first
    if (this.redis) {
      try {
        const key = `${this.mc.redisPrefix}:metrics:${expId}:history`;
        const raw = await this.redis.lrange(key, -window, -1);
        if (raw.length > 0) {
          return raw.map((s) => JSON.parse(s) as MetricPoint);
        }
      } catch {
        // Fall through to disk
      }
    }

    // Fallback: read from JSONL
    return this.readJsonlTail(expId, window);
  }

  /** Read full history from JSONL file, falling back to Redis. */
  async getFullHistory(expId: string): Promise<MetricPoint[]> {
    // Try disk first
    const filePath = path.join(
      this.experimentsDir,
      expId,
      this.mc.lmDirectory,
      this.mc.lmHistoryFile,
    );

    if (fs.existsSync(filePath)) {
      const points: MetricPoint[] = [];
      const stream = fs.createReadStream(filePath, { encoding: "utf-8" });
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

      for await (const line of rl) {
        if (line.trim()) {
          try {
            points.push(JSON.parse(line) as MetricPoint);
          } catch {
            // Skip malformed lines
          }
        }
      }

      if (points.length > 0) return points;
    }

    // Fallback: read full history from Redis list
    if (this.redis) {
      try {
        const key = `${this.mc.redisPrefix}:metrics:${expId}:history`;
        const raw = await this.redis.lrange(key, 0, -1);
        if (raw.length > 0) {
          return raw.map((s) => JSON.parse(s) as MetricPoint);
        }
      } catch {
        // No Redis data either
      }
    }

    return [];
  }

  /** Read training status. */
  async getStatus(expId: string): Promise<TrainingStatus | null> {
    if (this.redis) {
      try {
        const raw = await this.redis.get(`${this.mc.redisPrefix}:status:${expId}`);
        if (raw) return JSON.parse(raw) as TrainingStatus;
      } catch {
        // Fall through
      }
    }

    const filePath = path.join(
      this.experimentsDir,
      expId,
      this.mc.lmDirectory,
      this.mc.lmStatusFile,
    );
    try {
      const content = await fsp.readFile(filePath, "utf-8");
      return JSON.parse(content) as TrainingStatus;
    } catch {
      return null;
    }
  }

  /** Read latest metric point. */
  async getLatest(expId: string): Promise<MetricPoint | null> {
    if (this.redis) {
      try {
        const raw = await this.redis.get(`${this.mc.redisPrefix}:metrics:${expId}:latest`);
        if (raw) return JSON.parse(raw) as MetricPoint;
      } catch {
        // Fall through
      }
    }

    const filePath = path.join(
      this.experimentsDir,
      expId,
      this.mc.lmDirectory,
      this.mc.lmLatestFile,
    );
    try {
      const content = await fsp.readFile(filePath, "utf-8");
      return JSON.parse(content) as MetricPoint;
    } catch {
      return null;
    }
  }

  /** Read RL metrics. */
  async getRLMetrics(expId: string): Promise<RLTrainingMetrics | null> {
    if (this.redis) {
      try {
        const raw = await this.redis.get(`${this.mc.redisPrefix}:rl:${expId}:latest`);
        if (raw) return JSON.parse(raw) as RLTrainingMetrics;
      } catch {
        // Fall through
      }
    }

    // Fallback: RLTrainer writes to rl_metrics/rl_training_metrics.json
    const filePath = path.join(
      this.experimentsDir,
      expId,
      this.mc.rlDirectory,
      this.mc.rlMetricsFile,
    );
    try {
      const content = await fsp.readFile(filePath, "utf-8");
      return JSON.parse(content) as RLTrainingMetrics;
    } catch {
      return null;
    }
  }

  /** Read tail of JSONL file. */
  private async readJsonlTail(
    expId: string,
    limit: number,
  ): Promise<MetricPoint[]> {
    const filePath = path.join(
      this.experimentsDir,
      expId,
      this.mc.lmDirectory,
      this.mc.lmHistoryFile,
    );

    if (!fs.existsSync(filePath)) return [];

    const points: MetricPoint[] = [];
    const stream = fs.createReadStream(filePath, { encoding: "utf-8" });
    const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

    // Read all lines into a ring buffer
    const ring: string[] = [];
    for await (const line of rl) {
      if (line.trim()) {
        ring.push(line);
        if (ring.length > limit) ring.shift();
      }
    }

    for (const line of ring) {
      try {
        points.push(JSON.parse(line) as MetricPoint);
      } catch {
        // Skip
      }
    }

    return points;
  }
}
