import type { Database } from "./database.js";
import type { ObjectStore } from "./object-store.js";
import type { Report, ReportSummary, ReportVersion } from "@tidal/shared";
import type { FastifyBaseLogger } from "fastify";

export interface SaveResult {
  report: Report;
  spacesWritten: boolean;
  spacesError?: string;
}

const MAX_VERSIONS = 5;

/**
 * Repository pattern with write-through cache.
 *
 * - autoSave(id, patch): writes to SQLite only (session persistence)
 * - save(id, patch): writes to SQLite AND Spaces, creates version snapshot, prunes to 5
 * - load(id): SQLite first, then Spaces fallback
 * - listVersions(id): lists v{timestamp}.json keys from Spaces prefix, sorted desc
 * - restoreVersion(id, timestamp): fetches historical version from Spaces, updates SQLite
 *
 * When Spaces is not configured, save() behaves like autoSave() — graceful degradation.
 */
export class ReportRepository {
  private log?: FastifyBaseLogger;

  constructor(
    private db: Database,
    private store: ObjectStore,
    log?: FastifyBaseLogger,
  ) {
    this.log = log;
  }

  private currentKey(reportId: string): string {
    return `reports/${reportId}/current.json`;
  }

  private versionKey(reportId: string, timestamp: number): string {
    return `reports/${reportId}/v${timestamp}.json`;
  }

  private prefixKey(reportId: string): string {
    return `reports/${reportId}/`;
  }

  /** Create a new report (SQLite only). */
  create(title?: string, userId?: string): Report {
    return this.db.createReport(title, userId);
  }

  /** List all report summaries (SQLite only). */
  list(): ReportSummary[] {
    return this.db.listReports();
  }

  /**
   * Auto-save: writes to SQLite only. Used for debounced real-time persistence.
   * Does NOT write to Spaces.
   */
  autoSave(
    id: string,
    patch: { title?: string; blocks?: Record<string, unknown>[] },
  ): Report | null {
    return this.db.updateReport(id, patch);
  }

  /**
   * Explicit save: writes to SQLite and Spaces.
   * Creates a versioned snapshot and prunes versions beyond MAX_VERSIONS.
   * Spaces failure is non-fatal — SQLite write still succeeds, but the error
   * is reported via SaveResult so the caller can surface it.
   */
  async save(
    id: string,
    patch: { title?: string; blocks?: Record<string, unknown>[] },
  ): Promise<SaveResult | null> {
    // 1. Write to SQLite
    const report = this.db.updateReport(id, patch);
    if (!report) return null;

    // 2. Write to Spaces (best-effort)
    if (!this.store.isConfigured()) {
      return { report, spacesWritten: false };
    }

    try {
      const payload = JSON.stringify(report);
      const timestamp = report.updatedAt;

      // Write current.json
      await this.store.putObject(this.currentKey(id), payload, "application/json");

      // Write versioned snapshot
      await this.store.putObject(this.versionKey(id, timestamp), payload, "application/json");

      // Prune to MAX_VERSIONS
      await this.pruneVersions(id);

      return { report, spacesWritten: true };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.log?.error({ err, reportId: id }, "Failed to write report to Spaces: %s", message);
      return { report, spacesWritten: false, spacesError: message };
    }
  }

  /**
   * Load a report: SQLite first, then Spaces fallback.
   */
  async load(id: string): Promise<Report | null> {
    // 1. Try SQLite
    const fromDb = this.db.getReport(id);
    if (fromDb) return fromDb;

    // 2. Try Spaces
    if (!this.store.isConfigured()) return null;

    try {
      const raw = await this.store.getObject(this.currentKey(id));
      return JSON.parse(raw.toString()) as Report;
    } catch {
      return null;
    }
  }

  /**
   * List saved versions from Spaces, sorted by timestamp descending.
   * Returns empty array if Spaces is not configured.
   */
  async listVersions(reportId: string): Promise<ReportVersion[]> {
    if (!this.store.isConfigured()) return [];

    try {
      const keys = await this.store.listPrefix(this.prefixKey(reportId));
      const versions: ReportVersion[] = [];

      for (const key of keys) {
        // Match v{timestamp}.json pattern
        const match = key.match(/\/v(\d+)\.json$/);
        if (match) {
          versions.push({ timestamp: Number(match[1]), spacesKey: key });
        }
      }

      // Sort by timestamp descending (newest first)
      versions.sort((a, b) => b.timestamp - a.timestamp);
      return versions;
    } catch {
      return [];
    }
  }

  /**
   * Restore a historical version: fetches from Spaces, updates SQLite, writes new current.json.
   */
  async restoreVersion(reportId: string, timestamp: number): Promise<Report | null> {
    if (!this.store.isConfigured()) return null;

    try {
      const key = this.versionKey(reportId, timestamp);
      const raw = await this.store.getObject(key);
      const historical = JSON.parse(raw.toString()) as Report;

      // Update SQLite with historical content
      const restored = this.db.updateReport(reportId, {
        title: historical.title,
        blocks: historical.blocks,
      });
      if (!restored) return null;

      // Update current.json in Spaces
      try {
        await this.store.putObject(
          this.currentKey(reportId),
          JSON.stringify(restored),
          "application/json",
        );
      } catch {
        // Best-effort
      }

      return restored;
    } catch {
      return null;
    }
  }

  /**
   * Delete a report from SQLite and Spaces.
   */
  async delete(id: string): Promise<boolean> {
    const deleted = this.db.deleteReport(id);

    if (this.store.isConfigured()) {
      try {
        await this.store.deletePrefix(this.prefixKey(id));
      } catch {
        // Best-effort
      }
    }

    return deleted;
  }

  /**
   * Prune versioned snapshots to MAX_VERSIONS, deleting oldest first.
   */
  private async pruneVersions(reportId: string): Promise<void> {
    const versions = await this.listVersions(reportId);
    if (versions.length <= MAX_VERSIONS) return;

    // versions is sorted newest-first; delete from the tail
    const toDelete = versions.slice(MAX_VERSIONS);
    for (const v of toDelete) {
      await this.store.deleteObject(v.spacesKey).catch(() => {});
    }
  }
}
