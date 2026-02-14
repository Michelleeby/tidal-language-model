import fsp from "node:fs/promises";
import path from "node:path";
import { nanoid } from "nanoid";
import type { Report, ReportSummary, BlockContent } from "@tidal/shared";

/**
 * Filesystem-backed CRUD store for reports.
 * Each report is a standalone JSON file in `reportsDir`.
 */
export class ReportStore {
  constructor(private readonly reportsDir: string) {}

  /** Ensure the directory exists before any write. */
  private async ensureDir(): Promise<void> {
    await fsp.mkdir(this.reportsDir, { recursive: true });
  }

  private filePath(id: string): string {
    return path.join(this.reportsDir, `${id}.json`);
  }

  async create(title?: string): Promise<Report> {
    await this.ensureDir();

    const now = Date.now();
    const report: Report = {
      id: nanoid(),
      title: title ?? "Untitled Report",
      blocks: [],
      createdAt: now,
      updatedAt: now,
    };

    await fsp.writeFile(this.filePath(report.id), JSON.stringify(report, null, 2));
    return report;
  }

  async get(id: string): Promise<Report | null> {
    try {
      const raw = await fsp.readFile(this.filePath(id), "utf-8");
      return JSON.parse(raw) as Report;
    } catch {
      return null;
    }
  }

  async list(): Promise<ReportSummary[]> {
    await this.ensureDir();

    let files: string[];
    try {
      files = await fsp.readdir(this.reportsDir);
    } catch {
      return [];
    }

    const summaries: ReportSummary[] = [];

    for (const file of files) {
      if (!file.endsWith(".json")) continue;
      try {
        const raw = await fsp.readFile(path.join(this.reportsDir, file), "utf-8");
        const report = JSON.parse(raw) as Report;
        summaries.push({
          id: report.id,
          title: report.title,
          createdAt: report.createdAt,
          updatedAt: report.updatedAt,
        });
      } catch {
        // Skip malformed files
      }
    }

    // Sort by updatedAt descending (most recent first)
    summaries.sort((a, b) => b.updatedAt - a.updatedAt);
    return summaries;
  }

  async update(
    id: string,
    patch: { title?: string; blocks?: BlockContent[] },
  ): Promise<Report | null> {
    const report = await this.get(id);
    if (!report) return null;

    if (patch.title !== undefined) report.title = patch.title;
    if (patch.blocks !== undefined) report.blocks = patch.blocks;
    report.updatedAt = Date.now();

    await fsp.writeFile(this.filePath(id), JSON.stringify(report, null, 2));
    return report;
  }

  async delete(id: string): Promise<boolean> {
    try {
      await fsp.unlink(this.filePath(id));
      return true;
    } catch {
      return false;
    }
  }
}
