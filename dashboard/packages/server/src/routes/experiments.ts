import type { FastifyInstance } from "fastify";
import { ExperimentDiscovery } from "../services/experiment-discovery.js";
import type { ExperimentsResponse } from "@tidal/shared";

export default async function experimentsRoutes(fastify: FastifyInstance) {
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

  fastify.get<{ Reply: ExperimentsResponse }>(
    "/api/experiments",
    async () => {
      const experiments = await discovery.listExperiments();
      return { experiments };
    },
  );
}
