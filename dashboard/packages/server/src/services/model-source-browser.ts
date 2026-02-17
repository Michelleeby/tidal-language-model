import fsp from "node:fs/promises";
import path from "node:path";
import type { PluginFileNode } from "@tidal/shared";

const EXCLUDED_DIRS = new Set([
  "__pycache__",
  "data_cache",
  ".env",
  "tidal-env",
  "node_modules",
  "tests",
]);

const EXCLUDED_EXTENSIONS = new Set([".pyc"]);

/**
 * Read-only file browser for the tidal model source directory.
 */
export class ModelSourceBrowser {
  constructor(private sourceDir: string) {}

  /** Return a recursive file tree of the source directory. */
  async getFileTree(): Promise<PluginFileNode[]> {
    return this.walkDir(this.sourceDir, "");
  }

  /** Read a file's content. Throws on path traversal or missing files. */
  async readFile(relativePath: string): Promise<string> {
    this.validatePath(relativePath);
    const fullPath = path.join(this.sourceDir, relativePath);

    try {
      return await fsp.readFile(fullPath, "utf-8");
    } catch {
      throw new Error(`File not found: ${relativePath}`);
    }
  }

  private async walkDir(
    dir: string,
    prefix: string,
  ): Promise<PluginFileNode[]> {
    const entries = await fsp.readdir(dir, { withFileTypes: true });

    const dirs: PluginFileNode[] = [];
    const files: PluginFileNode[] = [];

    for (const entry of entries) {
      const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;

      if (entry.isDirectory()) {
        if (EXCLUDED_DIRS.has(entry.name)) continue;
        const children = await this.walkDir(
          path.join(dir, entry.name),
          relativePath,
        );
        dirs.push({
          name: entry.name,
          path: relativePath,
          type: "directory",
          children,
        });
      } else {
        if (EXCLUDED_EXTENSIONS.has(path.extname(entry.name))) continue;
        files.push({
          name: entry.name,
          path: relativePath,
          type: "file",
        });
      }
    }

    dirs.sort((a, b) => a.name.localeCompare(b.name));
    files.sort((a, b) => a.name.localeCompare(b.name));

    return [...dirs, ...files];
  }

  private validatePath(relativePath: string): void {
    if (path.isAbsolute(relativePath)) {
      throw new Error("Path traversal detected: absolute paths not allowed");
    }
    const resolved = path.resolve(this.sourceDir, relativePath);
    if (!resolved.startsWith(this.sourceDir + path.sep) && resolved !== this.sourceDir) {
      throw new Error("Path traversal detected: path resolves outside source directory");
    }
  }
}
