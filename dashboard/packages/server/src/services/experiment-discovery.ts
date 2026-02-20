import fsp from "node:fs/promises";
import path from "node:path";
import type Redis from "ioredis";
import type { ExperimentSummary, ExperimentType, TrainingStatus } from "@tidal/shared";

export interface ExperimentDiscoveryConfig {
  /** Redis key prefix, e.g. "tidal" */
  redisPrefix: string;
  /** LM metrics directory name, e.g. "dashboard_metrics" */
  lmDirectory: string;
  /** LM status filename, e.g. "status.json" */
  lmStatusFile: string;
}

const DEFAULT_CONFIG: ExperimentDiscoveryConfig = {
  redisPrefix: "tidal",
  lmDirectory: "dashboard_metrics",
  lmStatusFile: "status.json",
};

/**
 * Discovers experiments from Redis (set of IDs) or filesystem.
 */
/**
 * Convert a timestamp to milliseconds. Python's time.time() returns seconds
 * (~10 digits) while JS Date.now() returns milliseconds (~13 digits).
 * Timestamps below 10 billion are assumed to be in seconds.
 */
function toMilliseconds(ts: number): number {
  return ts < 1e10 ? ts * 1000 : ts;
}

export class ExperimentDiscovery {
  private dc: ExperimentDiscoveryConfig;

  constructor(
    private redis: Redis | null,
    private experimentsDir: string,
    config?: ExperimentDiscoveryConfig,
  ) {
    this.dc = config ?? DEFAULT_CONFIG;
  }

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
        const ids = await this.redis.smembers(`${this.dc.redisPrefix}:experiments`);
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
      const hasGpuInstance = entries.includes("gpu_instance.json");
      const checkpoints = entries.filter((f) => f.endsWith(".pth"));

      // Read status -- try disk first, then Redis
      let status: TrainingStatus | null = null;
      const statusPath = path.join(expPath, this.dc.lmDirectory, this.dc.lmStatusFile);
      try {
        const raw = await fsp.readFile(statusPath, "utf-8");
        status = JSON.parse(raw);
      } catch {
        // No status file on disk -- try Redis
        if (this.redis) {
          try {
            const raw = await this.redis.get(`${this.dc.redisPrefix}:status:${id}`);
            if (raw) status = JSON.parse(raw);
          } catch {
            // No Redis status either
          }
        }
      }

      // Read experiment metadata (type, lineage)
      let experimentType: ExperimentType = "unknown";
      let sourceExperimentId: string | null = null;
      let sourceCheckpoint: string | null = null;

      const metadataPath = path.join(expPath, "metadata.json");
      try {
        const metaRaw = await fsp.readFile(metadataPath, "utf-8");
        const meta = JSON.parse(metaRaw);
        if (meta.type === "lm" || meta.type === "rl") {
          experimentType = meta.type;
        }
        sourceExperimentId = meta.source_experiment_id ?? null;
        sourceCheckpoint = meta.source_checkpoint ?? null;
      } catch {
        // No metadata.json — infer type from directory contents
        const hasFoundational = entries.some((f) =>
          f.startsWith("checkpoint_foundational"),
        );
        const hasOnlyRL =
          !hasFoundational &&
          entries.some((f) => f.startsWith("rl_checkpoint"));
        experimentType = hasOnlyRL ? "rl" : hasFoundational ? "lm" : "unknown";
      }

      return {
        id,
        path: expPath,
        created: stat.birthtimeMs || stat.ctimeMs,
        hasRLMetrics,
        hasEvaluation,
        hasAblation,
        hasGpuInstance,
        status,
        checkpoints,
        experimentType,
        sourceExperimentId,
        sourceCheckpoint,
      };
    } catch {
      // No local directory -- try Redis (remote experiment)
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
      const statusRaw = await this.redis.get(`${this.dc.redisPrefix}:status:${id}`);
      const status: TrainingStatus | null = statusRaw
        ? JSON.parse(statusRaw)
        : null;

      // Only surface this experiment if we have at least status data in Redis
      if (!status) return null;

      const rawTs = status.start_time ?? status.last_update;
      return {
        id,
        path: "",
        created: toMilliseconds(rawTs),
        hasRLMetrics: false,
        hasEvaluation: false,
        hasAblation: false,
        hasGpuInstance: false,
        status,
        checkpoints: [],
        experimentType: "unknown",
        sourceExperimentId: null,
        sourceCheckpoint: null,
      };
    } catch {
      return null;
    }
  }
}
