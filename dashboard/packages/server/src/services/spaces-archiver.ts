import fsp from "node:fs/promises";
import path from "node:path";
import type { ObjectStore } from "./object-store.js";

export interface ArchiveManifest {
  state: "uploading" | "complete" | "failed";
  archivedAt: number;
  spacesPrefix: string;
  archivedFiles: string[];
  totalSizeBytes: number;
  localFilesRetained: string[];
}

/** Files/directories that are never uploaded to Spaces and always stay local. */
const LOCAL_ONLY = new Set([
  "_archive_manifest.json",
  "metadata.json",
  "config.yaml",
  "gpu_instance.json",
]);

/** Directory prefixes that always stay local (metrics for fast reads). */
const LOCAL_ONLY_DIRS = new Set(["dashboard_metrics", "rl_metrics"]);

/**
 * Two-phase archival of experiment checkpoints to DigitalOcean Spaces.
 *
 * Phase 1 — Upload: uploads all .pth files to Spaces, writes manifest with state:uploading
 * Phase 2 — Verify: HEAD-verifies each uploaded file, deletes local copies if verified
 * On failure at any stage: writes state:failed, preserves local files
 *
 * Idempotent: second call on state:complete is a no-op.
 */
export class SpacesArchiver {
  constructor(
    private store: ObjectStore,
    private experimentsDir: string,
  ) {}

  private manifestPath(expId: string): string {
    return path.join(this.experimentsDir, expId, "_archive_manifest.json");
  }

  private spacesKey(expId: string, filename: string): string {
    return `experiments/${expId}/${filename}`;
  }

  async getManifest(expId: string): Promise<ArchiveManifest | null> {
    try {
      const raw = await fsp.readFile(this.manifestPath(expId), "utf-8");
      return JSON.parse(raw) as ArchiveManifest;
    } catch {
      return null;
    }
  }

  async isArchived(expId: string): Promise<boolean> {
    const manifest = await this.getManifest(expId);
    return manifest?.state === "complete";
  }

  private async writeManifest(expId: string, manifest: ArchiveManifest): Promise<void> {
    await fsp.writeFile(this.manifestPath(expId), JSON.stringify(manifest, null, 2), "utf-8");
  }

  /**
   * Archive all .pth files from an experiment to Spaces.
   * Small metadata files stay local.
   */
  async archiveExperiment(expId: string): Promise<void> {
    const expDir = path.join(this.experimentsDir, expId);

    // Idempotency: already complete
    const existing = await this.getManifest(expId);
    if (existing?.state === "complete") return;

    // Collect .pth files to archive
    let entries: string[];
    try {
      entries = await fsp.readdir(expDir);
    } catch {
      return; // Directory doesn't exist
    }

    const pthFiles = entries.filter((f) => f.endsWith(".pth"));
    const spacesPrefix = `experiments/${expId}/`;

    // Write initial manifest with state:uploading
    const manifest: ArchiveManifest = {
      state: "uploading",
      archivedAt: Date.now(),
      spacesPrefix,
      archivedFiles: pthFiles,
      totalSizeBytes: 0,
      localFilesRetained: entries.filter((f) => !f.endsWith(".pth")),
    };
    await this.writeManifest(expId, manifest);

    try {
      let totalBytes = 0;

      // Upload each .pth file
      for (const filename of pthFiles) {
        const filePath = path.join(expDir, filename);
        const key = this.spacesKey(expId, filename);
        const stat = await fsp.stat(filePath);
        totalBytes += stat.size;
        await this.store.putLargeFile(key, filePath);
      }

      // HEAD-verify all uploaded files
      const failedVerification: string[] = [];
      for (const filename of pthFiles) {
        const key = this.spacesKey(expId, filename);
        const head = await this.store.headObject(key);
        if (!head.exists) {
          failedVerification.push(filename);
        }
      }

      if (failedVerification.length > 0) {
        // Verification failed — mark as failed, preserve local files
        manifest.state = "failed";
        await this.writeManifest(expId, manifest);
        return;
      }

      // All verified — delete local .pth files
      for (const filename of pthFiles) {
        const filePath = path.join(expDir, filename);
        await fsp.unlink(filePath).catch(() => {});
      }

      // Mark complete
      manifest.state = "complete";
      manifest.totalSizeBytes = totalBytes;
      await this.writeManifest(expId, manifest);
    } catch {
      // Upload failed — mark as failed, preserve local files
      manifest.state = "failed";
      await this.writeManifest(expId, manifest);
    }
  }

  /**
   * Download an archived file from Spaces back to the local experiment directory.
   */
  async retrieveFile(expId: string, filename: string): Promise<void> {
    const expDir = path.join(this.experimentsDir, expId);
    const key = this.spacesKey(expId, filename);
    const destPath = path.join(expDir, filename);

    await fsp.mkdir(expDir, { recursive: true });
    await this.store.downloadToFile(key, destPath);
  }
}
