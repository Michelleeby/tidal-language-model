import type { FastifyInstance } from "fastify";
import type {
  CreateJobRequest,
  CreateJobResponse,
  JobsListResponse,
  JobResponse,
  JobSignalRequest,
  JobSignalResponse,
} from "@tidal/shared";
import { JobStore } from "../services/job-store.js";
import { JobOrchestrator } from "../services/job-orchestrator.js";
import { ProvisioningChain } from "../services/provisioning-chain.js";
import { WorkerSpawner } from "../services/worker-spawner.js";
import { LocalProvider } from "../services/providers/local-provider.js";
import { AWSProvider } from "../services/providers/aws-provider.js";
import { VastAIProvider } from "../services/providers/vastai-provider.js";
import { ExperimentArchiver } from "../services/experiment-archiver.js";

export default async function jobRoutes(fastify: FastifyInstance) {
  const config = fastify.serverConfig;
  const store = new JobStore(fastify.redis);
  const spawner = new WorkerSpawner(
    config.projectRoot,
    config.pythonBin,
    config.redisUrl,
    fastify.log,
  );
  const chain = new ProvisioningChain([
    new LocalProvider(spawner),
    new AWSProvider(),
    new VastAIProvider({
      apiKey: config.vastaiApiKey,
      dashboardUrl: config.dashboardUrl,
      authToken: config.authToken,
      repoUrl: config.repoUrl,
      log: fastify.log,
    }),
  ]);
  const archiver = new ExperimentArchiver(
    fastify.redis,
    config.experimentsDir,
    fastify.log,
  );
  const orchestrator = new JobOrchestrator(
    store,
    chain,
    spawner,
    fastify.sseManager,
    fastify.log,
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
