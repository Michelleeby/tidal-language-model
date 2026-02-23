import fsp from "node:fs/promises";
import path from "node:path";
import type Redis from "ioredis";
import type { Database } from "./database.js";
import type { ObjectStore } from "./object-store.js";
import { JobStore } from "./job-store.js";

const STALENESS_THRESHOLD_S = 300; // 5 minutes
const REDIS_PREFIX = "tidal";

export interface DeleteResult {
  diskDeleted: boolean;
  redisKeysRemoved: number;
  analysesRemoved: number;
}

export interface CanDeleteResult {
  ok: boolean;
  reason?: string;
}

/**
 * Command pattern: cascading deletion of an experiment across disk, Redis, and SQLite.
 * Route guards against deleting actively-training experiments.
 */
export class ExperimentDeleter {
  private jobStore: JobStore;

  constructor(
    private redis: Redis | null,
    private experimentsDir: string,
    private db: Database,
    private objectStore?: ObjectStore | null,
  ) {
    this.jobStore = new JobStore(redis);
  }

  /**
   * Check whether an experiment can be safely deleted.
   * Returns ok=true if deletion is safe, ok=false with a reason otherwise.
   */
  async canDelete(expId: string): Promise<CanDeleteResult> {
    const expDir = path.join(this.experimentsDir, expId);

    // Non-existent experiments are safe to delete (idempotent)
    const dirExists = await fsp.access(expDir).then(() => true).catch(() => false);
    if (!dirExists) {
      return { ok: true };
    }

    // Check for in-progress archival
    const manifestPath = path.join(expDir, "_archive_manifest.json");
    try {
      const raw = await fsp.readFile(manifestPath, "utf-8");
      const manifest = JSON.parse(raw) as { state: string };
      if (manifest.state === "uploading") {
        return { ok: false, reason: "Archival in progress — cannot delete while archiving" };
      }
    } catch {
      // No manifest — not archiving
    }

    // Check status file for actively-training experiment
    const statusPath = path.join(expDir, "dashboard_metrics", "status.json");
    try {
      const raw = await fsp.readFile(statusPath, "utf-8");
      const status = JSON.parse(raw) as { status: string; last_update?: number };

      if (status.status === "training" && status.last_update) {
        const ageS = Date.now() / 1000 - status.last_update;
        if (ageS < STALENESS_THRESHOLD_S) {
          return {
            ok: false,
            reason: `Experiment appears to be actively training (last update ${Math.round(ageS)}s ago)`,
          };
        }
      }
    } catch {
      // No status file — treat as safe to delete
    }

    // Check for active jobs linked to this experiment
    try {
      const activeJobs = await this.jobStore.listActive();
      const linked = activeJobs.find((j) => j.experimentId === expId);
      if (linked) {
        return { ok: false, reason: `Active job linked to experiment: ${linked.jobId}` };
      }
    } catch {
      // Redis unavailable — skip job check
    }

    return { ok: true };
  }

  /**
   * Delete an experiment across disk, Redis, and SQLite.
   * Idempotent — safe to call multiple times.
   */
  async delete(expId: string): Promise<DeleteResult> {
    let diskDeleted = false;
    let redisKeysRemoved = 0;
    let analysesRemoved = 0;

    const expDir = path.join(this.experimentsDir, expId);
    const dirExisted = await fsp.access(expDir).then(() => true).catch(() => false);

    // 0. Delete from Spaces if archived (read manifest before deleting disk)
    if (dirExisted && this.objectStore?.isConfigured()) {
      const manifestPath = path.join(expDir, "_archive_manifest.json");
      try {
        const raw = await fsp.readFile(manifestPath, "utf-8");
        const manifest = JSON.parse(raw) as { state: string; spacesPrefix: string };
        if (manifest.state === "complete" && manifest.spacesPrefix) {
          await this.objectStore.deletePrefix(manifest.spacesPrefix).catch(() => {});
        }
      } catch {
        // No manifest or not archived
      }
    }

    // 1. Delete from disk
    if (dirExisted) {
      try {
        await fsp.rm(expDir, { recursive: true, force: true });
        diskDeleted = true;
      } catch {
        diskDeleted = false;
      }
    }

    // 2. Delete Redis keys
    if (this.redis) {
      try {
        const prefix = REDIS_PREFIX;
        const keysToDelete = [
          `${prefix}:status:${expId}`,
          `${prefix}:metrics:${expId}:history`,
          `${prefix}:metrics:${expId}:latest`,
          `${prefix}:rl:${expId}:latest`,
        ];
        const deleted = await this.redis.del(...keysToDelete);
        await this.redis.srem(`${prefix}:experiments`, expId);
        redisKeysRemoved = typeof deleted === "number" ? deleted : 0;
      } catch {
        // Redis unavailable — continue
      }
    }

    // 3. Delete SQLite analysis results
    analysesRemoved = this.db.deleteAnalysesByExperiment(expId);

    return { diskDeleted, redisKeysRemoved, analysesRemoved };
  }
}
