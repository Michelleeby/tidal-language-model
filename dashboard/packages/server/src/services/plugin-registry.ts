import fsp from "node:fs/promises";
import path from "node:path";
import { parse as parseYaml } from "yaml";
import type {
  PluginManifest,
  TrainingPhase,
} from "@tidal/shared";

/** Minimal logger interface — compatible with Fastify's pino logger. */
export interface RegistryLogger {
  info(msg: string): void;
  warn(msg: string): void;
}

/** No-op logger used when no logger is provided. */
const nullLogger: RegistryLogger = {
  info() {},
  warn() {},
};

/**
 * Scans a plugins directory for manifest.yaml files and provides
 * typed access to plugin manifests.
 */
export class PluginRegistry {
  private plugins = new Map<string, PluginManifest>();
  private log: RegistryLogger;

  constructor(
    private pluginsDir: string,
    logger?: RegistryLogger,
  ) {
    this.log = logger ?? nullLogger;
  }

  /** Scan the plugins directory and load all valid manifests. */
  async load(): Promise<void> {
    this.plugins.clear();

    let entries: string[];
    try {
      const dirents = await fsp.readdir(this.pluginsDir, {
        withFileTypes: true,
      });
      entries = dirents.filter((d) => d.isDirectory()).map((d) => d.name);
    } catch {
      this.log.warn(`Plugins directory not found: ${this.pluginsDir}`);
      return;
    }

    for (const dirName of entries) {
      const manifestPath = path.join(this.pluginsDir, dirName, "manifest.yaml");
      try {
        const raw = await fsp.readFile(manifestPath, "utf-8");
        const parsed = parseYaml(raw) as Record<string, unknown>;
        const manifest = this.validate(parsed);
        if (manifest) {
          this.plugins.set(manifest.name, manifest);
          this.log.info(
            `Loaded plugin: ${manifest.name} v${manifest.version}`,
          );
        } else {
          this.log.warn(
            `Plugin manifest validation failed for '${dirName}' — missing required fields`,
          );
        }
      } catch (err) {
        this.log.warn(
          `Failed to parse manifest in '${dirName}': ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }

    this.log.info(
      `Plugin registry: ${this.plugins.size} plugin(s) loaded`,
    );
  }

  /** Get a plugin by name. */
  get(name: string): PluginManifest | undefined {
    return this.plugins.get(name);
  }

  /** Get the default plugin (first loaded, or undefined). */
  getDefault(): PluginManifest | undefined {
    const first = this.plugins.values().next();
    return first.done ? undefined : first.value;
  }

  /** List all loaded plugins. */
  list(): PluginManifest[] {
    return [...this.plugins.values()];
  }

  /** Get a specific training phase from a plugin. */
  getPhase(pluginName: string, phaseId: string): TrainingPhase | undefined {
    const plugin = this.plugins.get(pluginName);
    if (!plugin) return undefined;
    return plugin.trainingPhases.find((p) => p.id === phaseId);
  }

  /** Validate parsed YAML has all required fields. Returns null if invalid. */
  private validate(raw: Record<string, unknown>): PluginManifest | null {
    if (
      typeof raw.name !== "string" ||
      typeof raw.displayName !== "string" ||
      typeof raw.version !== "string" ||
      typeof raw.description !== "string" ||
      !Array.isArray(raw.trainingPhases) ||
      raw.trainingPhases.length === 0 ||
      !Array.isArray(raw.checkpointPatterns) ||
      !raw.generation ||
      !raw.metrics ||
      !raw.redis ||
      !raw.infrastructure
    ) {
      return null;
    }

    // Type-assert after validation — the YAML structure matches our interfaces
    return raw as unknown as PluginManifest;
  }
}
