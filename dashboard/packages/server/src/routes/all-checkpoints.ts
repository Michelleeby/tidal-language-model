import fsp from "node:fs/promises";
import path from "node:path";
import type { FastifyInstance } from "fastify";
import type { AllCheckpointsResponse, CheckpointInfo } from "@tidal/shared";
import { classifyCheckpoint } from "../services/checkpoint-classifier.js";
import { ExperimentDiscovery } from "../services/experiment-discovery.js";

const RL_ELIGIBLE_PHASES = new Set(["foundational", "final"]);

export default async function allCheckpointsRoutes(fastify: FastifyInstance) {
  const plugin = fastify.tidalManifest;
  const discoveryConfig = plugin
    ? {
        redisPrefix: plugin.metrics.redisPrefix,
        lmDirectory: plugin.metrics.lm.directory,
        lmStatusFile: plugin.metrics.lm.statusFile,
      }
    : undefined;

  const discovery = new ExperimentDiscovery(
    fastify.redis,
    fastify.serverConfig.experimentsDir,
    discoveryConfig,
  );

  fastify.get<{ Reply: AllCheckpointsResponse }>(
    "/api/checkpoints",
    async (request) => {
      const experiments = await discovery.listExperiments();
      const patterns = plugin?.checkpointPatterns ?? [];

      const groups: AllCheckpointsResponse["groups"] = [];

      for (const exp of experiments) {
        // Only include LM experiments (or unknown legacy experiments that have checkpoints)
        if (exp.experimentType === "rl") continue;
        if (!exp.path) continue;

        const checkpoints: CheckpointInfo[] = [];
        try {
          const entries = await fsp.readdir(exp.path);
          for (const filename of entries) {
            if (!filename.endsWith(".pth")) continue;
            const filePath = path.join(exp.path, filename);
            const stat = await fsp.stat(filePath);
            const { phase, epoch } = classifyCheckpoint(filename, patterns);

            if (!RL_ELIGIBLE_PHASES.has(phase)) continue;

            checkpoints.push({
              filename,
              path: filePath,
              sizeBytes: stat.size,
              modified: stat.mtimeMs,
              phase,
              epoch,
            });
          }
        } catch {
          continue;
        }

        if (checkpoints.length > 0) {
          checkpoints.sort((a, b) => a.modified - b.modified);
          groups.push({ experimentId: exp.id, checkpoints });
        }
      }

      // Sort groups by most recent experiment first
      groups.sort((a, b) => {
        const expA = experiments.find((e) => e.id === a.experimentId);
        const expB = experiments.find((e) => e.id === b.experimentId);
        return (expB?.created ?? 0) - (expA?.created ?? 0);
      });

      return { groups };
    },
  );
}
