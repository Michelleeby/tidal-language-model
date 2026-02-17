import fsp from "node:fs/promises";
import { parse as parseYaml } from "yaml";
import type { PluginManifest } from "@tidal/shared";

/** Minimal logger interface — compatible with Fastify's pino logger. */
export interface ManifestLogger {
  info(msg: string): void;
  warn(msg: string): void;
}

const nullLogger: ManifestLogger = { info() {}, warn() {} };

/**
 * Load and validate the tidal manifest from a specific file path.
 * Returns the typed manifest or null on any failure.
 */
export async function loadTidalManifest(
  manifestPath: string,
  logger?: ManifestLogger,
): Promise<PluginManifest | null> {
  const log = logger ?? nullLogger;

  let raw: string;
  try {
    raw = await fsp.readFile(manifestPath, "utf-8");
  } catch {
    log.warn(`Failed to read tidal manifest: ${manifestPath}`);
    return null;
  }

  let parsed: Record<string, unknown>;
  try {
    parsed = parseYaml(raw) as Record<string, unknown>;
  } catch (err) {
    log.warn(
      `Failed to parse tidal manifest YAML: ${err instanceof Error ? err.message : String(err)}`,
    );
    return null;
  }

  if (
    typeof parsed.name !== "string" ||
    typeof parsed.displayName !== "string" ||
    typeof parsed.version !== "string" ||
    typeof parsed.description !== "string" ||
    !Array.isArray(parsed.trainingPhases) ||
    parsed.trainingPhases.length === 0 ||
    !Array.isArray(parsed.checkpointPatterns) ||
    !parsed.generation ||
    !parsed.metrics ||
    !parsed.redis ||
    !parsed.infrastructure
  ) {
    log.warn(`Tidal manifest validation failed — missing required fields`);
    return null;
  }

  return parsed as unknown as PluginManifest;
}
