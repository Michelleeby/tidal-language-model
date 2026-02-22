import type { FastifyInstance } from "fastify";
import { ExperimentDiscovery } from "../services/experiment-discovery.js";
import { ExperimentDeleter } from "../services/experiment-deleter.js";
import { SpacesArchiver } from "../services/spaces-archiver.js";
import type {
  ExperimentsResponse,
  DeleteExperimentResponse,
  ArchiveExperimentResponse,
  RetrieveCheckpointResponse,
} from "@tidal/shared";

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
    fastify.objectStore,
  );

  const archiver = new SpacesArchiver(
    fastify.objectStore,
    fastify.serverConfig.experimentsDir,
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

  fastify.post<{ Params: { expId: string }; Reply: ArchiveExperimentResponse }>(
    "/api/experiments/:expId/archive",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const { expId } = request.params;

      if (!fastify.objectStore.isConfigured()) {
        return reply.status(503).send({ error: "Object storage not configured" } as any);
      }

      const existing = await archiver.getManifest(expId);
      if (existing?.state === "complete") {
        return { expId, state: "already_archived" };
      }

      await archiver.archiveExperiment(expId);
      const manifest = await archiver.getManifest(expId);
      return { expId, state: (manifest?.state as "complete" | "failed") ?? "failed" };
    },
  );

  fastify.post<{ Params: { expId: string; filename: string }; Reply: RetrieveCheckpointResponse }>(
    "/api/experiments/:expId/retrieve/:filename",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const { expId, filename } = request.params;

      if (!fastify.objectStore.isConfigured()) {
        return reply.status(503).send({ error: "Object storage not configured" } as any);
      }

      await archiver.retrieveFile(expId, filename);
      return { expId, filename };
    },
  );
}
