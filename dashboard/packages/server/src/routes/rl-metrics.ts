import type { FastifyInstance } from "fastify";
import { MetricsReader, metricsReaderConfigFromManifest } from "../services/metrics-reader.js";
import type { RLMetricsResponse } from "@tidal/shared";

export default async function rlMetricsRoutes(fastify: FastifyInstance) {
  const plugin = fastify.pluginRegistry.getDefault();
  const metricsConfig = plugin
    ? metricsReaderConfigFromManifest(plugin.metrics)
    : undefined;

  const reader = new MetricsReader(
    fastify.redis,
    fastify.serverConfig.experimentsDir,
    metricsConfig,
  );

  fastify.get<{
    Params: { expId: string };
    Reply: RLMetricsResponse;
  }>("/api/experiments/:expId/rl-metrics", async (request) => {
    const { expId } = request.params;
    const metrics = await reader.getRLMetrics(expId);
    return { expId, metrics };
  });
}
