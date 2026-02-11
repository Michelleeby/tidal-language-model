import type { FastifyInstance } from "fastify";
import { MetricsReader } from "../services/metrics-reader.js";
import { lttbDownsample } from "../services/downsampler.js";
import type { MetricsResponse } from "@tidal/shared";

export default async function metricsRoutes(fastify: FastifyInstance) {
  const reader = new MetricsReader(
    fastify.redis,
    fastify.serverConfig.experimentsDir,
  );

  fastify.get<{
    Params: { expId: string };
    Querystring: { mode?: string; window?: string; maxPoints?: string };
    Reply: MetricsResponse;
  }>("/api/experiments/:expId/metrics", async (request) => {
    const { expId } = request.params;
    const mode = request.query.mode ?? "recent";
    const window = parseInt(request.query.window ?? "5000", 10);
    const maxPoints = parseInt(request.query.maxPoints ?? "2000", 10);

    let points;
    let downsampled = false;
    let originalCount: number;

    if (mode === "recent") {
      points = await reader.getRecentMetrics(expId, window);
      originalCount = points.length;
    } else {
      points = await reader.getFullHistory(expId);
      originalCount = points.length;
      if (points.length > maxPoints) {
        points = lttbDownsample(points, maxPoints);
        downsampled = true;
      }
    }

    return {
      expId,
      points,
      totalPoints: points.length,
      originalCount,
      downsampled,
    };
  });
}
