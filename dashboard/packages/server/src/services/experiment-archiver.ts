import type { FastifyBaseLogger } from "fastify";
import type { Redis } from "ioredis";
import * as fsp from "node:fs/promises";
import * as path from "node:path";

/**
 * Archives experiment data from Redis to disk so it survives
 * after Redis TTLs expire (e.g. when a remote vast.ai instance
 * is deprovisioned).
 *
 * Writes files in the exact structure that ExperimentDiscovery
 * and MetricsReader expect on disk.
 */
export class ExperimentArchiver {
  constructor(
    private redis: Redis | null,
    private experimentsDir: string,
    private log: FastifyBaseLogger,
  ) {}

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

    const expDir = path.join(this.experimentsDir, expId);
    const dashboardDir = path.join(expDir, "dashboard_metrics");
    const rlDir = path.join(expDir, "rl_metrics");

    await fsp.mkdir(dashboardDir, { recursive: true });

    let archived = 0;

    // 1. Metrics history list → metrics.jsonl
    try {
      const history = await this.redis.lrange(
        `tidal:metrics:${expId}:history`,
        0,
        -1,
      );
      if (history.length > 0) {
        const jsonl = history.map((line) => line.trimEnd()).join("\n") + "\n";
        await fsp.writeFile(
          path.join(dashboardDir, "metrics.jsonl"),
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

    // 2. Status → status.json
    try {
      const status = await this.redis.get(`tidal:status:${expId}`);
      if (status) {
        await fsp.writeFile(
          path.join(dashboardDir, "status.json"),
          status,
          "utf-8",
        );
        archived++;
      }
    } catch (err) {
      this.log.error({ expId, err }, "Failed to archive status");
    }

    // 3. Latest metrics → latest.json
    try {
      const latest = await this.redis.get(`tidal:metrics:${expId}:latest`);
      if (latest) {
        await fsp.writeFile(
          path.join(dashboardDir, "latest.json"),
          latest,
          "utf-8",
        );
        archived++;
      }
    } catch (err) {
      this.log.error({ expId, err }, "Failed to archive latest metrics");
    }

    // 4. RL metrics → rl_training_metrics.json
    try {
      const rl = await this.redis.get(`tidal:rl:${expId}:latest`);
      if (rl) {
        await fsp.mkdir(rlDir, { recursive: true });
        await fsp.writeFile(
          path.join(rlDir, "rl_training_metrics.json"),
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
