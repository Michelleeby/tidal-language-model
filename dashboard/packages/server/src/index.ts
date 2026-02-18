import Fastify from "fastify";
import path from "node:path";
import fastifyStatic from "@fastify/static";
import cookie from "@fastify/cookie";
import { loadConfig, validateConfig, type ServerConfig } from "./config.js";
import type { PluginManifest } from "@tidal/shared";
import { loadTidalManifest } from "./services/tidal-manifest-loader.js";
import { migrateLegacyReports } from "./services/report-migration.js";
import redisPlugin from "./plugins/redis.js";
import corsPlugin from "./plugins/cors.js";
import ssePlugin from "./plugins/sse.js";
import authPlugin from "./plugins/auth.js";
import databasePlugin from "./plugins/database.js";
import rateLimitPlugin from "./plugins/rate-limit.js";
import provisioningPlugin from "./plugins/provisioning.js";
import experimentsRoutes from "./routes/experiments.js";
import metricsRoutes from "./routes/metrics.js";
import rlMetricsRoutes from "./routes/rl-metrics.js";
import statusRoutes from "./routes/status.js";
import checkpointsRoutes from "./routes/checkpoints.js";
import evaluationRoutes from "./routes/evaluation.js";
import gpuInstanceRoutes from "./routes/gpu-instance.js";
import generateRoutes from "./routes/generate.js";
import sseRoutes from "./routes/sse.js";
import jobRoutes from "./routes/jobs.js";
import workerRoutes from "./routes/workers.js";
import pluginsRoutes from "./routes/plugins.js";
import configsRoutes from "./routes/configs.js";
import reportsRoutes from "./routes/reports.js";
import authRoutes from "./routes/auth.js";
import modelSourceRoutes from "./routes/model-source.js";
import analyzeRoutes from "./routes/analyze.js";

declare module "fastify" {
  interface FastifyInstance {
    serverConfig: ServerConfig;
    tidalManifest: PluginManifest | null;
  }
}

async function main() {
  const config = loadConfig();

  // Validate config before constructing the server
  const issues = validateConfig(config);
  for (const issue of issues) {
    if (issue.level === "error") {
      console.error(`Config error: ${issue.message}`);
    } else {
      console.warn(`Config warning: ${issue.message}`);
    }
  }
  if (issues.some((i) => i.level === "error")) {
    process.exit(1);
  }

  const fastify = Fastify({
    bodyLimit: 5 * 1024 * 1024, // 5MB — defense in depth for large worker payloads
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

  // Load tidal manifest directly
  const manifestPath = path.join(config.projectRoot, "plugins", "tidal", "manifest.yaml");
  const tidalManifest = await loadTidalManifest(manifestPath, fastify.log);
  if (!tidalManifest) {
    fastify.log.warn(
      "Tidal manifest not found — checkpoint classification and generation will not work",
    );
  }
  fastify.decorate("tidalManifest", tidalManifest);

  // Plugins (order matters: cookie + database before auth, redis before sse, all before routes)
  await fastify.register(cookie);
  await fastify.register(corsPlugin);
  await fastify.register(databasePlugin);
  await fastify.register(redisPlugin, { url: config.redisUrl });
  await fastify.register(ssePlugin);
  await fastify.register(authPlugin);
  await fastify.register(rateLimitPlugin);
  await fastify.register(provisioningPlugin);

  // Migrate legacy JSON reports into SQLite (idempotent, runs before routes)
  const reportsDir = path.join(config.projectRoot, "reports");
  const migrationResult = await migrateLegacyReports(
    reportsDir,
    fastify.db,
    fastify.log,
  );
  if (migrationResult.imported > 0) {
    fastify.log.info(
      `Report migration: ${migrationResult.imported} imported, ${migrationResult.skipped} skipped, ${migrationResult.errors} errors`,
    );
  }

  // API routes
  await fastify.register(authRoutes);
  await fastify.register(experimentsRoutes);
  await fastify.register(metricsRoutes);
  await fastify.register(rlMetricsRoutes);
  await fastify.register(statusRoutes);
  await fastify.register(checkpointsRoutes);
  await fastify.register(evaluationRoutes);
  await fastify.register(gpuInstanceRoutes);
  await fastify.register(generateRoutes);
  await fastify.register(sseRoutes);
  await fastify.register(jobRoutes);
  await fastify.register(workerRoutes);
  await fastify.register(pluginsRoutes);
  await fastify.register(configsRoutes);
  await fastify.register(reportsRoutes);
  await fastify.register(modelSourceRoutes);
  await fastify.register(analyzeRoutes);

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
