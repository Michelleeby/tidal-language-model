import type { FastifyInstance } from "fastify";
import type { PluginsListResponse, PluginResponse } from "@tidal/shared";

export default async function pluginsRoutes(fastify: FastifyInstance) {
  /** GET /api/plugins — list all plugins (summary). */
  fastify.get<{ Reply: PluginsListResponse }>(
    "/api/plugins",
    async () => {
      const plugins = fastify.pluginRegistry.list().map((p) => ({
        name: p.name,
        displayName: p.displayName,
        version: p.version,
        trainingPhases: p.trainingPhases.map((tp) => ({
          id: tp.id,
          displayName: tp.displayName,
        })),
        generationModes: p.generation.modes.map((m) => ({
          id: m.id,
          displayName: m.displayName,
        })),
      }));
      return { plugins };
    },
  );

  /** GET /api/plugins/:name — full manifest for a specific plugin. */
  fastify.get<{ Params: { name: string }; Reply: PluginResponse }>(
    "/api/plugins/:name",
    async (request, reply) => {
      const plugin = fastify.pluginRegistry.get(request.params.name);
      if (!plugin) {
        return reply.status(404).send({ plugin: null });
      }
      return { plugin };
    },
  );
}
