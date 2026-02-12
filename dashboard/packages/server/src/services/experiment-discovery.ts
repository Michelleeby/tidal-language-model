import fsp from "node:fs/promises";
import path from "node:path";
import type Redis from "ioredis";
import type { ExperimentSummary, TrainingStatus } from "@tidal/shared";

/**
 * Discovers experiments from Redis (set of IDs) or filesystem.
 */
export class ExperimentDiscovery {
  constructor(
    private redis: Redis | null,
    private experimentsDir: string,
  ) {}

  async listExperiments(): Promise<ExperimentSummary[]> {
    const ids = await this.getExperimentIds();
    const summaries: ExperimentSummary[] = [];

    for (const id of ids) {
      const summary = await this.getExperimentSummary(id);
      if (summary) summaries.push(summary);
    }

    // Sort by creation time descending (newest first)
    summaries.sort((a, b) => b.created - a.created);
    return summaries;
  }

  private async getExperimentIds(): Promise<string[]> {
    // Try Redis first
    if (this.redis) {
      try {
        const ids = await this.redis.smembers("tidal:experiments");
        if (ids.length > 0) return ids;
      } catch {
        // Fall through
      }
    }

    // Fallback: scan experiments directory
    try {
      const entries = await fsp.readdir(this.experimentsDir, {
        withFileTypes: true,
      });
      return entries
        .filter((e) => e.isDirectory())
        .map((e) => e.name);
    } catch {
      return [];
    }
  }

  private async getExperimentSummary(
    id: string,
  ): Promise<ExperimentSummary | null> {
    const expPath = path.join(this.experimentsDir, id);

    // Try local filesystem first (local experiments have full metadata)
    try {
      const stat = await fsp.stat(expPath);
      if (!stat.isDirectory()) return null;

      const entries = await fsp.readdir(expPath);

      // Check for various result files
      const hasRLMetrics =
        entries.includes("rl_metrics") ||
        entries.some((f) => f.startsWith("rl_checkpoint"));
      const hasEvaluation = entries.includes("evaluation_results.json");
      const hasAblation = entries.includes("ablation_results.json");
      const checkpoints = entries.filter((f) => f.endsWith(".pth"));

      // Read status
      let status: TrainingStatus | null = null;
      const statusPath = path.join(expPath, "dashboard_metrics", "status.json");
      try {
        const raw = await fsp.readFile(statusPath, "utf-8");
        status = JSON.parse(raw);
      } catch {
        // No status file
      }

      return {
        id,
        path: expPath,
        created: stat.birthtimeMs || stat.ctimeMs,
        hasRLMetrics,
        hasEvaluation,
        hasAblation,
        status,
        checkpoints,
      };
    } catch {
      // No local directory — try Redis (remote experiment)
      return this.getRedisOnlySummary(id);
    }
  }

  /**
   * Build a summary for experiments that only exist in Redis (remote workers).
   * Status and metrics live in Redis keys, not on the local filesystem.
   */
  private async getRedisOnlySummary(
    id: string,
  ): Promise<ExperimentSummary | null> {
    if (!this.redis) return null;

    try {
      const statusRaw = await this.redis.get(`tidal:status:${id}`);
      const status: TrainingStatus | null = statusRaw
        ? JSON.parse(statusRaw)
        : null;

      // Only surface this experiment if we have at least status data in Redis
      if (!status) return null;

      return {
        id,
        path: "",
        created: status.start_time ?? status.last_update,
        hasRLMetrics: false,
        hasEvaluation: false,
        hasAblation: false,
        status,
        checkpoints: [],
      };
    } catch {
      return null;
    }
  }
}
