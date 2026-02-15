import fsp from "node:fs/promises";
import path from "node:path";
import type { Database } from "./database.js";

interface MigrationLog {
  info: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  error: (...args: unknown[]) => void;
}

export interface MigrationResult {
  imported: number;
  skipped: number;
  errors: number;
}

/**
 * Reads legacy JSON report files from `reportsDir` and imports them into the
 * SQLite database with `user_id = NULL`. Uses INSERT OR IGNORE so running
 * multiple times is safe (idempotent).
 */
export async function migrateLegacyReports(
  reportsDir: string,
  db: Database,
  log: MigrationLog,
): Promise<MigrationResult> {
  const result: MigrationResult = { imported: 0, skipped: 0, errors: 0 };

  let files: string[];
  try {
    files = await fsp.readdir(reportsDir);
  } catch {
    // Directory doesn't exist — nothing to migrate
    return result;
  }

  for (const file of files) {
    if (!file.endsWith(".json")) continue;

    try {
      const raw = await fsp.readFile(path.join(reportsDir, file), "utf-8");
      const data = JSON.parse(raw);

      const imported = db.importLegacyReport({
        id: data.id,
        title: data.title ?? "Untitled Report",
        blocks: data.blocks ?? [],
        createdAt: data.createdAt ?? Date.now(),
        updatedAt: data.updatedAt ?? Date.now(),
      });

      if (imported) {
        result.imported++;
        log.info(`Migrated report: ${data.id}`);
      } else {
        result.skipped++;
      }
    } catch (err) {
      result.errors++;
      log.warn(`Failed to migrate ${file}: ${err}`);
    }
  }

  return result;
}
