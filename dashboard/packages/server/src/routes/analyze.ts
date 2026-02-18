import type { FastifyInstance } from "fastify";
import { GenerationBridge } from "../services/generation-bridge.js";
import type { AnalyzeRequest, AnalyzeResponse } from "@tidal/shared";

export default async function analyzeRoutes(fastify: FastifyInstance) {
  const plugin = fastify.tidalManifest;
  const bridge = new GenerationBridge(fastify.serverConfig, plugin ?? undefined);

  fastify.post<{
    Body: AnalyzeRequest;
  }>("/api/analyze-trajectories", { preHandler: [fastify.rateLimit] }, async (request, reply) => {
    if (!bridge.available) {
      return reply.status(503).send({
        error:
          "Trajectory analysis is not available on this host — " +
          "no inference service found.",
      });
    }

    try {
      return await bridge.analyzeTrajectories(request.body);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      if (
        message.includes("not available") ||
        message.includes("Inference sidecar") ||
        message.includes("requires the inference")
      ) {
        return reply.status(503).send({ error: message });
      }
      throw err;
    }
  });
}
