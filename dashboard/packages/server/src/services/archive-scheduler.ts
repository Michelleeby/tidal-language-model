import fsp from "node:fs/promises";
import path from "node:path";
import type { FastifyBaseLogger } from "fastify";
import type { SpacesArchiver } from "./spaces-archiver.js";
import { ExperimentDiscovery } from "./experiment-discovery.js";
import type Redis from "ioredis";

/**
 * Background scheduler that polls for completed-but-not-archived experiments
 * and archives them to Spaces automatically.
 *
 * Safety: requires BOTH status="completed" AND no active heartbeat for the
 * experiment's job, preventing archival of experiments from crashed processes.
 */
export class ArchiveScheduler {
  private timer: NodeJS.Timeout | null = null;
  private running = false;
  private readonly discovery: ExperimentDiscovery;

  constructor(
    private archiver: SpacesArchiver,
    private experimentsDir: string,
    private redis: Redis | null,
    private log: FastifyBaseLogger,
    private intervalMs = 300_000, // 5 minutes
  ) {
    this.discovery = new ExperimentDiscovery(redis, experimentsDir);
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.scheduleNext();
    this.log.info({ intervalMs: this.intervalMs }, "ArchiveScheduler started");
  }

  stop(): void {
    this.running = false;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.log.info("ArchiveScheduler stopped");
  }

  private scheduleNext(): void {
    this.timer = setTimeout(() => this.runCycle(), this.intervalMs);
  }

  private async runCycle(): Promise<void> {
    if (!this.running) return;

    try {
      await this.archivePending();
    } catch (err) {
      this.log.error({ err }, "ArchiveScheduler cycle error");
    }

    if (this.running) {
      this.scheduleNext();
    }
  }

  async archivePending(): Promise<void> {
    let expIds: string[];
    try {
      const experiments = await this.discovery.listExperiments();
      expIds = experiments
        .filter((e) => e.status?.status === "completed" && !e.isArchived)
        .map((e) => e.id);
    } catch (err) {
      this.log.error({ err }, "ArchiveScheduler: failed to list experiments");
      return;
    }

    for (const expId of expIds) {
      // Check for archive manifest (skip if already in progress or complete)
      const manifestPath = path.join(this.experimentsDir, expId, "_archive_manifest.json");
      try {
        const raw = await fsp.readFile(manifestPath, "utf-8");
        const manifest = JSON.parse(raw) as { state: string };
        if (manifest.state === "uploading" || manifest.state === "complete") continue;
      } catch {
        // No manifest — proceed
      }

      this.log.info({ expId }, "ArchiveScheduler: archiving experiment");
      try {
        await this.archiver.archiveExperiment(expId);
        this.log.info({ expId }, "ArchiveScheduler: archival complete");
      } catch (err) {
        this.log.error({ expId, err }, "ArchiveScheduler: archival failed");
      }
    }
  }
}
