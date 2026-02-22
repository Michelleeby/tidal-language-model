import type { FastifyInstance } from "fastify";
import { ExperimentDiscovery } from "../services/experiment-discovery.js";
import { ExperimentDeleter } from "../services/experiment-deleter.js";
import type { ExperimentsResponse, DeleteExperimentResponse } from "@tidal/shared";

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

  const deleter = new ExperimentDeleter(
    fastify.redis,
    fastify.serverConfig.experimentsDir,
    fastify.db,
  );

  fastify.get<{ Reply: ExperimentsResponse }>(
    "/api/experiments",
    async () => {
      const experiments = await discovery.listExperiments();
      return { experiments };
    },
  );

  fastify.delete<{ Params: { expId: string }; Reply: DeleteExperimentResponse }>(
    "/api/experiments/:expId",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const { expId } = request.params;

      const check = await deleter.canDelete(expId);
      if (!check.ok) {
        return reply.status(409).send({ error: check.reason } as any);
      }

      const result = await deleter.delete(expId);
      return result;
    },
  );
}
