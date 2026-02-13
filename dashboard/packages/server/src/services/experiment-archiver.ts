import type { FastifyBaseLogger } from "fastify";
import type { Redis } from "ioredis";
import * as fsp from "node:fs/promises";
import * as path from "node:path";
import type { MetricsConfig } from "@tidal/shared";

export interface ArchiverConfig {
  redisPrefix: string;
  lmDirectory: string;
  lmHistoryFile: string;
  lmStatusFile: string;
  lmLatestFile: string;
  rlDirectory: string;
  rlMetricsFile: string;
}

const DEFAULT_CONFIG: ArchiverConfig = {
  redisPrefix: "tidal",
  lmDirectory: "dashboard_metrics",
  lmHistoryFile: "metrics.jsonl",
  lmStatusFile: "status.json",
  lmLatestFile: "latest.json",
  rlDirectory: "rl_metrics",
  rlMetricsFile: "rl_training_metrics.json",
};

export function archiverConfigFromManifest(metrics: MetricsConfig): ArchiverConfig {
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

/**
 * Archives experiment data from Redis to disk so it survives
 * after Redis TTLs expire (e.g. when a remote vast.ai instance
 * is deprovisioned).
 *
 * Writes files in the exact structure that ExperimentDiscovery
 * and MetricsReader expect on disk.
 */
export class ExperimentArchiver {
  private ac: ArchiverConfig;

  constructor(
    private redis: Redis | null,
    private experimentsDir: string,
    private log: FastifyBaseLogger,
    config?: ArchiverConfig,
  ) {
    this.ac = config ?? DEFAULT_CONFIG;
  }

  /**
   * Archive all Redis data for an experiment to disk.
   * Idempotent — safe to call multiple times.
   */
  async archive(expId: string): Promise<void> {
    if (!this.redis) {
      this.log.warn({ expId }, "Cannot archive — Redis unavailable");
      return;
    }

    if (!expId) {
      this.log.warn("Cannot archive — no experiment ID");
      return;
    }

    const prefix = this.ac.redisPrefix;
    const expDir = path.join(this.experimentsDir, expId);
    const dashboardDir = path.join(expDir, this.ac.lmDirectory);
    const rlDir = path.join(expDir, this.ac.rlDirectory);

    await fsp.mkdir(dashboardDir, { recursive: true });

    let archived = 0;

    // 1. Metrics history list -> JSONL
    try {
      const history = await this.redis.lrange(
        `${prefix}:metrics:${expId}:history`,
        0,
        -1,
      );
      if (history.length > 0) {
        const jsonl = history.map((line) => line.trimEnd()).join("\n") + "\n";
        await fsp.writeFile(
          path.join(dashboardDir, this.ac.lmHistoryFile),
          jsonl,
          "utf-8",
        );
        archived++;
        this.log.info(
          { expId, points: history.length },
          "Archived metrics history",
        );
      }
    } catch (err) {
      this.log.error({ expId, err }, "Failed to archive metrics history");
    }

    // 2. Status
    try {
      const status = await this.redis.get(`${prefix}:status:${expId}`);
      if (status) {
        await fsp.writeFile(
          path.join(dashboardDir, this.ac.lmStatusFile),
          status,
          "utf-8",
        );
        archived++;
      }
    } catch (err) {
      this.log.error({ expId, err }, "Failed to archive status");
    }

    // 3. Latest metrics
    try {
      const latest = await this.redis.get(`${prefix}:metrics:${expId}:latest`);
      if (latest) {
        await fsp.writeFile(
          path.join(dashboardDir, this.ac.lmLatestFile),
          latest,
          "utf-8",
        );
        archived++;
      }
    } catch (err) {
      this.log.error({ expId, err }, "Failed to archive latest metrics");
    }

    // 4. RL metrics
    try {
      const rl = await this.redis.get(`${prefix}:rl:${expId}:latest`);
      if (rl) {
        await fsp.mkdir(rlDir, { recursive: true });
        await fsp.writeFile(
          path.join(rlDir, this.ac.rlMetricsFile),
          rl,
          "utf-8",
        );
        archived++;
      }
    } catch (err) {
      this.log.error({ expId, err }, "Failed to archive RL metrics");
    }

    this.log.info({ expId, archived }, "Experiment archival complete");
  }
}
