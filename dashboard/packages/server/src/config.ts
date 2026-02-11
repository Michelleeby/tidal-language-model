import path from "node:path";

import type { ComputeProviderType } from "@tidal/shared";

export interface ServerConfig {
  port: number;
  host: string;
  redisUrl: string;
  experimentsDir: string;
  pythonBin: string;
  projectRoot: string;
  defaultComputeProvider: ComputeProviderType;
  authToken: string | null;
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
    defaultComputeProvider:
      (process.env.DEFAULT_COMPUTE_PROVIDER as ComputeProviderType) ?? "local",
    authToken: process.env.TIDAL_AUTH_TOKEN ?? null,
  };
}
