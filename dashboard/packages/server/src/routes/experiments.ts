import type { FastifyInstance } from "fastify";
import { ExperimentDiscovery } from "../services/experiment-discovery.js";
import type { ExperimentsResponse } from "@tidal/shared";

export default async function experimentsRoutes(fastify: FastifyInstance) {
  const discovery = new ExperimentDiscovery(
    fastify.redis,
    fastify.serverConfig.experimentsDir,
  );

  fastify.get<{ Reply: ExperimentsResponse }>(
    "/api/experiments",
    async () => {
      const experiments = await discovery.listExperiments();
      return { experiments };
    },
  );
}
