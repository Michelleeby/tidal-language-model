import type { FastifyInstance } from "fastify";
import type {
  CreateJobRequest,
  CreateJobResponse,
  JobsListResponse,
  JobResponse,
  JobSignalRequest,
  JobSignalResponse,
  JobLogsResponse,
} from "@tidal/shared";
import { JobStore, jobStoreKeysFromManifest } from "../services/job-store.js";
import { JobOrchestrator } from "../services/job-orchestrator.js";
import { ExperimentArchiver, archiverConfigFromManifest } from "../services/experiment-archiver.js";
import { JobPolicyRegistry } from "../services/job-policy.js";

export default async function jobRoutes(fastify: FastifyInstance) {
  const config = fastify.serverConfig;
  const plugin = fastify.pluginRegistry.getDefault();
  const storeKeys = plugin
    ? jobStoreKeysFromManifest(plugin.redis)
    : undefined;
  const store = new JobStore(fastify.redis, undefined, storeKeys);
  const archiverConf = plugin
    ? archiverConfigFromManifest(plugin.metrics)
    : undefined;
  const archiver = new ExperimentArchiver(
    fastify.redis,
    config.experimentsDir,
    fastify.log,
    archiverConf,
  );
  const policyRegistry = new JobPolicyRegistry(fastify.pluginRegistry);
  const orchestrator = new JobOrchestrator(
    store,
    fastify.provisioningChain,
    fastify.workerSpawner,
    fastify.sseManager,
    fastify.log,
    policyRegistry,
    { defaultProvider: config.defaultComputeProvider },
    archiver,
  );

  fastify.addHook("onClose", () => orchestrator.stop());

  // POST /api/jobs — create a new training job
  fastify.post<{ Body: CreateJobRequest }>("/api/jobs", { preHandler: [fastify.verifyAuth] }, async (request, reply) => {
    try {
      const job = await orchestrator.createJob(request.body);
      return reply.status(201).send({ job } satisfies CreateJobResponse);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      if (message.includes("already running")) {
        return reply.status(409).send({ error: message });
      }
      if (message.includes("Redis unavailable")) {
        return reply.status(503).send({ error: message });
      }
      throw err;
    }
  });

  // GET /api/jobs — list all jobs
  fastify.get("/api/jobs", async (_request, reply) => {
    try {
      const jobs = await store.list();
      return reply.send({ jobs } satisfies JobsListResponse);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      if (message.includes("Redis unavailable")) {
        return reply.status(503).send({ error: message });
      }
      throw err;
    }
  });

  // GET /api/jobs/active — get the currently active job
  fastify.get("/api/jobs/active", async (_request, reply) => {
    try {
      const job = await orchestrator.getActiveJob();
      return reply.send({ job } satisfies JobResponse);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      if (message.includes("Redis unavailable")) {
        return reply.status(503).send({ error: message });
      }
      throw err;
    }
  });

  // GET /api/jobs/:jobId — get a specific job
  fastify.get<{ Params: { jobId: string } }>(
    "/api/jobs/:jobId",
    async (request, reply) => {
      try {
        const job = await store.get(request.params.jobId);
        return reply.send({ job } satisfies JobResponse);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        if (message.includes("Redis unavailable")) {
          return reply.status(503).send({ error: message });
        }
        throw err;
      }
    },
  );

  // GET /api/jobs/:jobId/logs — retrieve stored log lines
  fastify.get<{
    Params: { jobId: string };
    Querystring: { offset?: string; limit?: string };
  }>(
    "/api/jobs/:jobId/logs",
    async (request, reply) => {
      const { jobId } = request.params;
      const redis = fastify.redis;
      if (!redis) {
        return reply.status(503).send({ error: "Redis unavailable" });
      }

      const offset = parseInt(request.query.offset ?? "0", 10);
      const limit = parseInt(request.query.limit ?? "5000", 10);
      const totalLines = await redis.llen(`tidal:logs:${jobId}`);

      const end = offset + limit - 1;
      const rawLines: string[] = await redis.lrange(
        `tidal:logs:${jobId}`,
        offset,
        end,
      );
      const lines = rawLines.map((raw) => JSON.parse(raw));

      return reply.send({
        jobId,
        lines,
        totalLines,
      } satisfies JobLogsResponse);
    },
  );

  // POST /api/jobs/:jobId/signal — send a signal to a running job
  fastify.post<{ Params: { jobId: string }; Body: JobSignalRequest }>(
    "/api/jobs/:jobId/signal",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      try {
        const job = await orchestrator.signalJob(
          request.params.jobId,
          request.body.signal,
        );
        return reply.send({
          ok: true,
          status: job.status,
        } satisfies JobSignalResponse);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        if (message.includes("not found")) {
          return reply.status(404).send({ error: message });
        }
        if (message.includes("Cannot signal")) {
          return reply.status(409).send({ error: message });
        }
        if (message.includes("Redis unavailable")) {
          return reply.status(503).send({ error: message });
        }
        throw err;
      }
    },
  );

  // POST /api/jobs/:jobId/cancel — cancel a job in any non-terminal state
  fastify.post<{ Params: { jobId: string } }>(
    "/api/jobs/:jobId/cancel",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      try {
        const job = await orchestrator.cancelJob(request.params.jobId);
        return reply.send({
          ok: true,
          status: job.status,
        } satisfies JobSignalResponse);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        if (message.includes("not found")) {
          return reply.status(404).send({ error: message });
        }
        if (message.includes("already in terminal")) {
          return reply.status(409).send({ error: message });
        }
        if (message.includes("Redis unavailable")) {
          return reply.status(503).send({ error: message });
        }
        throw err;
      }
    },
  );
}
