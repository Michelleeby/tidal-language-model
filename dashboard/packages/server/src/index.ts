import Fastify from "fastify";
import path from "node:path";
import fastifyStatic from "@fastify/static";
import { loadConfig, type ServerConfig } from "./config.js";
import redisPlugin from "./plugins/redis.js";
import corsPlugin from "./plugins/cors.js";
import ssePlugin from "./plugins/sse.js";
import authPlugin from "./plugins/auth.js";
import rateLimitPlugin from "./plugins/rate-limit.js";
import experimentsRoutes from "./routes/experiments.js";
import metricsRoutes from "./routes/metrics.js";
import rlMetricsRoutes from "./routes/rl-metrics.js";
import statusRoutes from "./routes/status.js";
import checkpointsRoutes from "./routes/checkpoints.js";
import evaluationRoutes from "./routes/evaluation.js";
import generateRoutes from "./routes/generate.js";
import sseRoutes from "./routes/sse.js";
import jobRoutes from "./routes/jobs.js";
import workerRoutes from "./routes/workers.js";

declare module "fastify" {
  interface FastifyInstance {
    serverConfig: ServerConfig;
  }
}

async function main() {
  const config = loadConfig();

  const fastify = Fastify({
    logger: {
      level: "info",
      transport: {
        target: "pino-pretty",
        options: { translateTime: "HH:MM:ss Z", ignore: "pid,hostname" },
      },
    },
  });

  // Decorate with config
  fastify.decorate("serverConfig", config);

  // Plugins (order matters: redis before sse, auth/rate-limit after redis, all before routes)
  await fastify.register(corsPlugin);
  await fastify.register(redisPlugin, { url: config.redisUrl });
  await fastify.register(ssePlugin);
  await fastify.register(authPlugin);
  await fastify.register(rateLimitPlugin);

  // API routes
  await fastify.register(experimentsRoutes);
  await fastify.register(metricsRoutes);
  await fastify.register(rlMetricsRoutes);
  await fastify.register(statusRoutes);
  await fastify.register(checkpointsRoutes);
  await fastify.register(evaluationRoutes);
  await fastify.register(generateRoutes);
  await fastify.register(sseRoutes);
  await fastify.register(jobRoutes);
  await fastify.register(workerRoutes);

  // Serve built client in production
  const clientDist = path.resolve(
    import.meta.dirname,
    "..",
    "..",
    "client",
    "dist",
  );
  await fastify.register(fastifyStatic, {
    root: clientDist,
    wildcard: false,
  });

  // SPA fallback: serve index.html for non-API routes
  fastify.setNotFoundHandler(async (request, reply) => {
    if (request.url.startsWith("/api/")) {
      return reply.status(404).send({ error: "Not found" });
    }
    return reply.sendFile("index.html");
  });

  // Start
  await fastify.listen({ port: config.port, host: config.host });
  fastify.log.info(
    `Dashboard server listening on http://localhost:${config.port}`,
  );
}

main().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
