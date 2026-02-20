/**
 * Pre-creates an experiment directory and ID before the training subprocess
 * starts, so the dashboard can navigate to the experiment immediately.
 */

import { execFileSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { Redis } from "ioredis";

function getGitShortHash(cwd: string): string {
  try {
    return execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd })
      .toString()
      .trim();
  } catch {
    return "nogit";
  }
}

function formatTimestamp(): string {
  const now = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-` +
    `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`
  );
}

/**
 * Derive the source experiment ID from a checkpoint path like
 * "experiments/<exp-id>/model.pth".
 */
function deriveSourceExperimentId(checkpointPath: string): string | null {
  const parts = checkpointPath.replace(/\\/g, "/").split("/");
  const idx = parts.indexOf("experiments");
  if (idx !== -1 && idx + 1 < parts.length) {
    return parts[idx + 1];
  }
  return null;
}

export interface PreCreateConfig {
  configPath?: string;
  checkpoint?: string;
  resumeExpDir?: string;
  rlConfigPath?: string;
}

/**
 * Pre-create an experiment so the dashboard can navigate to it immediately.
 *
 * For resume jobs (config.resumeExpDir), extracts the existing experiment ID
 * and skips directory creation.
 *
 * @returns The experiment ID (new or existing).
 */
export async function preCreateExperiment(
  jobType: string,
  config: PreCreateConfig,
  experimentsDir: string,
  redis: Pick<Redis, "sadd">,
  projectRoot: string,
): Promise<string> {
  // Resume: extract existing experiment ID from the resume path
  if (config.resumeExpDir) {
    const parts = config.resumeExpDir.replace(/\\/g, "/").split("/");
    const idx = parts.indexOf("experiments");
    const experimentId = idx !== -1 && idx + 1 < parts.length
      ? parts[idx + 1]
      : parts[parts.length - 1];

    await redis.sadd("tidal:experiments", experimentId);
    return experimentId;
  }

  // Generate experiment ID matching Python format
  const timestamp = formatTimestamp();
  const gitHash = getGitShortHash(projectRoot);
  const randomHex = crypto.randomBytes(5).toString("hex");
  const isRL = jobType.includes("rl");
  const typeTag = isRL ? "rl" : "config";
  const experimentId = `${timestamp}-commit_${gitHash}-${typeTag}_${randomHex}`;

  // Create experiment directory
  const expDir = path.join(experimentsDir, experimentId);
  fs.mkdirSync(expDir, { recursive: true });

  // Derive source experiment for RL jobs
  const sourceExperimentId = isRL && config.checkpoint
    ? deriveSourceExperimentId(config.checkpoint)
    : null;
  const sourceCheckpoint = isRL && config.checkpoint
    ? config.checkpoint
    : null;

  // Write metadata.json
  const metadata = {
    type: isRL ? "rl" : "lm",
    created_at: new Date().toISOString(),
    source_experiment_id: sourceExperimentId,
    source_checkpoint: sourceCheckpoint,
  };
  fs.writeFileSync(
    path.join(expDir, "metadata.json"),
    JSON.stringify(metadata, null, 2),
  );

  // Register in Redis so SSE and experiments list pick it up
  await redis.sadd("tidal:experiments", experimentId);

  return experimentId;
}
