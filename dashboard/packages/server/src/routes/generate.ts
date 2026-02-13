import type { FastifyInstance } from "fastify";
import { GenerationBridge } from "../services/generation-bridge.js";
import type { GenerateRequest, GenerateResponse } from "@tidal/shared";

export default async function generateRoutes(fastify: FastifyInstance) {
  const plugin = fastify.pluginRegistry.getDefault();
  const bridge = new GenerationBridge(fastify.serverConfig, plugin);

  fastify.post<{
    Body: GenerateRequest;
  }>("/api/generate", { preHandler: [fastify.rateLimit] }, async (request, reply) => {
    if (!bridge.available) {
      return reply.status(503).send({
        error:
          "Text generation is not available on this host — " +
          "no inference service or Python environment found.",
      });
    }

    try {
      return await bridge.generate(request.body);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      if (
        message.includes("not available") ||
        message.includes("Inference sidecar")
      ) {
        return reply.status(503).send({ error: message });
      }
      throw err;
    }
  });
}
