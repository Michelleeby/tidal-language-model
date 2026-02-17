import type { FastifyInstance } from "fastify";
import type { PluginsListResponse, PluginResponse } from "@tidal/shared";

export default async function pluginsRoutes(fastify: FastifyInstance) {
  /** GET /api/plugins — list plugins (single tidal manifest). */
  fastify.get<{ Reply: PluginsListResponse }>(
    "/api/plugins",
    async () => {
      const manifest = fastify.tidalManifest;
      if (!manifest) return { plugins: [] };

      return {
        plugins: [
          {
            name: manifest.name,
            displayName: manifest.displayName,
            version: manifest.version,
            trainingPhases: manifest.trainingPhases.map((tp) => ({
              id: tp.id,
              displayName: tp.displayName,
            })),
            generationModes: manifest.generation.modes.map((m) => ({
              id: m.id,
              displayName: m.displayName,
            })),
          },
        ],
      };
    },
  );

  /** GET /api/plugins/:name — full manifest for a specific plugin. */
  fastify.get<{ Params: { name: string }; Reply: PluginResponse }>(
    "/api/plugins/:name",
    async (request, reply) => {
      const manifest = fastify.tidalManifest;
      if (!manifest || manifest.name !== request.params.name) {
        return reply.status(404).send({ plugin: null });
      }
      return { plugin: manifest };
    },
  );
}
