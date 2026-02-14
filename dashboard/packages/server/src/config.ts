import path from "node:path";

import type { ComputeProviderType } from "@tidal/shared";

export interface ConfigWarning {
  level: "warn" | "error";
  message: string;
}

/**
 * Validate server config at startup. Returns a list of warnings/errors.
 * Callers should log warnings and throw on errors.
 */
export function validateConfig(config: ServerConfig): ConfigWarning[] {
  const issues: ConfigWarning[] = [];

  if (!config.authToken) {
    issues.push({
      level: "warn",
      message: "TIDAL_AUTH_TOKEN is not set — worker auth and remote jobs will not work",
    });
  }

  if (config.defaultComputeProvider === "vastai") {
    if (!config.vastaiApiKey) {
      issues.push({
        level: "error",
        message: "DEFAULT_COMPUTE_PROVIDER is vastai but VASTAI_API_KEY is not set",
      });
    }
    if (!config.dashboardUrl) {
      issues.push({
        level: "error",
        message: "DEFAULT_COMPUTE_PROVIDER is vastai but TIDAL_DASHBOARD_URL is not set",
      });
    }
    if (!config.repoUrl) {
      issues.push({
        level: "error",
        message: "DEFAULT_COMPUTE_PROVIDER is vastai but TIDAL_REPO_URL is not set",
      });
    }
  }

  if (config.defaultComputeProvider === "digitalocean") {
    if (!config.digitaloceanApiKey) {
      issues.push({
        level: "error",
        message: "DEFAULT_COMPUTE_PROVIDER is digitalocean but DO_API_KEY is not set",
      });
    }
    if (!config.dashboardUrl) {
      issues.push({
        level: "error",
        message: "DEFAULT_COMPUTE_PROVIDER is digitalocean but TIDAL_DASHBOARD_URL is not set",
      });
    }
    if (!config.repoUrl) {
      issues.push({
        level: "error",
        message: "DEFAULT_COMPUTE_PROVIDER is digitalocean but TIDAL_REPO_URL is not set",
      });
    }
  }

  return issues;
}

export interface ServerConfig {
  port: number;
  host: string;
  redisUrl: string;
  experimentsDir: string;
  pythonBin: string;
  projectRoot: string;
  inferenceUrl: string | null;
  defaultComputeProvider: ComputeProviderType;
  authToken: string | null;
  vastaiApiKey: string | null;
  digitaloceanApiKey: string | null;
  digitaloceanRegion: string;
  repoUrl: string | null;
  dashboardUrl: string | null;
}

export function loadConfig(): ServerConfig {
  const projectRoot = path.resolve(
    import.meta.dirname,
    "..",
    "..",
    "..",
    "..",
  );

  return {
    port: parseInt(process.env.PORT ?? "4400", 10),
    host: process.env.HOST ?? "0.0.0.0",
    redisUrl: process.env.REDIS_URL ?? "redis://localhost:6379",
    experimentsDir:
      process.env.EXPERIMENTS_DIR ?? path.join(projectRoot, "experiments"),
    pythonBin:
      process.env.PYTHON_BIN ??
      path.join(projectRoot, "tidal-env", "bin", "python"),
    projectRoot,
    inferenceUrl: process.env.INFERENCE_URL ?? null,
    defaultComputeProvider:
      (process.env.DEFAULT_COMPUTE_PROVIDER as ComputeProviderType) ?? "local",
    authToken: process.env.TIDAL_AUTH_TOKEN ?? null,
    vastaiApiKey: process.env.VASTAI_API_KEY ?? null,
    digitaloceanApiKey: process.env.DO_API_KEY ?? null,
    digitaloceanRegion: process.env.DO_REGION ?? "tor1",
    repoUrl: process.env.TIDAL_REPO_URL ?? null,
    dashboardUrl: process.env.TIDAL_DASHBOARD_URL ?? null,
  };
}
