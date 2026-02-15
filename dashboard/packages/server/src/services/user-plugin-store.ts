import fsp from "node:fs/promises";
import path from "node:path";
import type { PluginFileNode } from "@tidal/shared";

const ALLOWED_EXTENSIONS = new Set([
  ".py",
  ".yaml",
  ".yml",
  ".txt",
  ".md",
  ".json",
  ".cfg",
]);

const TEMPLATE_EXCLUDES = new Set([
  "__pycache__",
  "data_cache",
  ".env",
]);

const MAX_FILE_SIZE = 1024 * 1024; // 1MB

export class UserPluginStore {
  constructor(
    private userPluginsDir: string,
    private templateDir: string,
  ) {}

  // ---------------------------------------------------------------------------
  // Template copy
  // ---------------------------------------------------------------------------

  async createFromTemplate(
    userId: string,
    pluginName: string,
  ): Promise<void> {
    const destDir = path.join(this.userPluginsDir, userId, pluginName);
    await fsp.mkdir(destDir, { recursive: true });
    await this.copyDir(this.templateDir, destDir);
  }

  private async copyDir(src: string, dest: string): Promise<void> {
    const entries = await fsp.readdir(src, { withFileTypes: true });

    for (const entry of entries) {
      if (TEMPLATE_EXCLUDES.has(entry.name)) continue;
      if (entry.name.endsWith(".pyc")) continue;

      const srcPath = path.join(src, entry.name);
      const destPath = path.join(dest, entry.name);

      if (entry.isDirectory()) {
        await fsp.mkdir(destPath, { recursive: true });
        await this.copyDir(srcPath, destPath);
      } else {
        await fsp.copyFile(srcPath, destPath);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // File tree
  // ---------------------------------------------------------------------------

  async getFileTree(
    userId: string,
    pluginName: string,
  ): Promise<PluginFileNode[]> {
    const pluginDir = path.join(this.userPluginsDir, userId, pluginName);
    return this.walkDir(pluginDir, "");
  }

  private async walkDir(
    baseDir: string,
    relativePath: string,
  ): Promise<PluginFileNode[]> {
    const fullDir = relativePath
      ? path.join(baseDir, relativePath)
      : baseDir;
    const entries = await fsp.readdir(fullDir, { withFileTypes: true });
    const nodes: PluginFileNode[] = [];

    for (const entry of entries) {
      const entryRelPath = relativePath
        ? `${relativePath}/${entry.name}`
        : entry.name;

      if (entry.isDirectory()) {
        const children = await this.walkDir(baseDir, entryRelPath);
        nodes.push({
          name: entry.name,
          path: entryRelPath,
          type: "directory",
          children,
        });
      } else {
        nodes.push({
          name: entry.name,
          path: entryRelPath,
          type: "file",
        });
      }
    }

    return nodes.sort((a, b) => {
      // Directories first, then alphabetical
      if (a.type !== b.type) return a.type === "directory" ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
  }

  // ---------------------------------------------------------------------------
  // Read file
  // ---------------------------------------------------------------------------

  async readFile(
    userId: string,
    pluginName: string,
    relativePath: string,
  ): Promise<string> {
    const filePath = this.validatePath(userId, pluginName, relativePath);
    return fsp.readFile(filePath, "utf-8");
  }

  // ---------------------------------------------------------------------------
  // Write file
  // ---------------------------------------------------------------------------

  async writeFile(
    userId: string,
    pluginName: string,
    relativePath: string,
    content: string,
  ): Promise<void> {
    if (content.length > MAX_FILE_SIZE) {
      throw new Error("File exceeds maximum size of 1MB");
    }

    const ext = path.extname(relativePath).toLowerCase();
    if (!ALLOWED_EXTENSIONS.has(ext)) {
      throw new Error(
        `Extension "${ext}" is not allowed. Allowed: ${[...ALLOWED_EXTENSIONS].join(", ")}`,
      );
    }

    const filePath = this.validatePath(userId, pluginName, relativePath);

    // Ensure parent directory exists (for new nested files)
    await fsp.mkdir(path.dirname(filePath), { recursive: true });
    await fsp.writeFile(filePath, content, "utf-8");
  }

  // ---------------------------------------------------------------------------
  // Delete file
  // ---------------------------------------------------------------------------

  async deleteFile(
    userId: string,
    pluginName: string,
    relativePath: string,
  ): Promise<void> {
    if (path.basename(relativePath) === "manifest.yaml") {
      throw new Error("Cannot delete manifest.yaml — it is required");
    }

    const filePath = this.validatePath(userId, pluginName, relativePath);
    await fsp.unlink(filePath);
  }

  // ---------------------------------------------------------------------------
  // Delete entire plugin
  // ---------------------------------------------------------------------------

  async deletePlugin(userId: string, pluginName: string): Promise<void> {
    const pluginDir = path.join(this.userPluginsDir, userId, pluginName);
    await fsp.rm(pluginDir, { recursive: true, force: true });
  }

  // ---------------------------------------------------------------------------
  // Path validation — single entry point for all traversal checks
  // ---------------------------------------------------------------------------

  private validatePath(
    userId: string,
    pluginName: string,
    relativePath: string,
  ): string {
    const pluginDir = path.resolve(
      this.userPluginsDir,
      userId,
      pluginName,
    );
    const resolved = path.resolve(pluginDir, relativePath);

    if (!resolved.startsWith(pluginDir + path.sep) && resolved !== pluginDir) {
      throw new Error("Invalid path: traversal outside plugin directory");
    }

    return resolved;
  }
}
