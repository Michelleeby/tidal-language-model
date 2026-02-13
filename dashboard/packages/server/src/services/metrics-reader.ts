import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import readline from "node:readline";
import type Redis from "ioredis";
import type { MetricPoint, TrainingStatus, RLTrainingMetrics } from "@tidal/shared";

/**
 * Reads metrics from Redis (primary) or JSONL files (fallback).
 */
export class MetricsReader {
  constructor(
    private redis: Redis | null,
    private experimentsDir: string,
  ) {}

  /** Read recent metrics (last `window` points). */
  async getRecentMetrics(
    expId: string,
    window: number,
  ): Promise<MetricPoint[]> {
    // Try Redis first
    if (this.redis) {
      try {
        const key = `tidal:metrics:${expId}:history`;
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
      "dashboard_metrics",
      "metrics.jsonl",
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
        const key = `tidal:metrics:${expId}:history`;
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
        const raw = await this.redis.get(`tidal:status:${expId}`);
        if (raw) return JSON.parse(raw) as TrainingStatus;
      } catch {
        // Fall through
      }
    }

    const filePath = path.join(
      this.experimentsDir,
      expId,
      "dashboard_metrics",
      "status.json",
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
        const raw = await this.redis.get(`tidal:metrics:${expId}:latest`);
        if (raw) return JSON.parse(raw) as MetricPoint;
      } catch {
        // Fall through
      }
    }

    const filePath = path.join(
      this.experimentsDir,
      expId,
      "dashboard_metrics",
      "latest.json",
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
        const raw = await this.redis.get(`tidal:rl:${expId}:latest`);
        if (raw) return JSON.parse(raw) as RLTrainingMetrics;
      } catch {
        // Fall through
      }
    }

    // Fallback: RLTrainer writes to rl_metrics/rl_training_metrics.json
    const filePath = path.join(
      this.experimentsDir,
      expId,
      "rl_metrics",
      "rl_training_metrics.json",
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
      "dashboard_metrics",
      "metrics.jsonl",
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
