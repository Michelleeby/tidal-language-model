import type { FastifyInstance } from "fastify";
import { MetricsReader } from "../services/metrics-reader.js";
import type { StatusResponse } from "@tidal/shared";

export default async function statusRoutes(fastify: FastifyInstance) {
  const reader = new MetricsReader(
    fastify.redis,
    fastify.serverConfig.experimentsDir,
  );

  fastify.get<{
    Params: { expId: string };
    Reply: StatusResponse;
  }>("/api/experiments/:expId/status", async (request) => {
    const { expId } = request.params;
    const status = await reader.getStatus(expId);
    return { expId, status };
  });
}
