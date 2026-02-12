import type { FastifyInstance } from "fastify";
import type { JobSignal, JobStatus } from "@tidal/shared";
import { JobStore } from "../services/job-store.js";
import type { SSEManager } from "../services/sse-manager.js";

export default async function workerRoutes(fastify: FastifyInstance) {
  const store = new JobStore(fastify.redis);
  const sseManager: SSEManager = fastify.sseManager;

  // POST /api/workers/:jobId/heartbeat — worker sends heartbeat
  fastify.post<{ Params: { jobId: string } }>(
    "/api/workers/:jobId/heartbeat",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const { jobId } = request.params;
      const job = await store.get(jobId);
      if (!job) {
        return reply.status(404).send({ error: "Job not found" });
      }
      await store.setHeartbeat(jobId);
      return reply.send({ ok: true });
    },
  );

  // PATCH /api/workers/:jobId/status — worker updates job status
  fastify.patch<{
    Params: { jobId: string };
    Body: { status: JobStatus; error?: string };
  }>(
    "/api/workers/:jobId/status",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const { jobId } = request.params;
      const { status, error } = request.body;

      const patch: Record<string, unknown> = { status };
      if (error) patch.error = error;
      if (status === "running") patch.startedAt = Date.now();
      if (status === "completed" || status === "failed") {
        patch.completedAt = Date.now();
      }

      const updated = await store.update(jobId, patch);
      if (!updated) {
        return reply.status(404).send({ error: "Job not found" });
      }

      sseManager.broadcastJobUpdate(updated);
      return reply.send({ ok: true, status: updated.status });
    },
  );

  // POST /api/workers/:jobId/metrics — worker forwards training metrics
  fastify.post<{
    Params: { jobId: string };
    Body: {
      expId: string;
      points?: Array<Record<string, unknown>>;
      status?: Record<string, unknown>;
    };
  }>(
    "/api/workers/:jobId/metrics",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const redis = fastify.redis;
      if (!redis) {
        return reply.status(503).send({ error: "Redis unavailable" });
      }

      const { expId, points, status } = request.body;
      if (!expId) {
        return reply.status(400).send({ error: "expId is required" });
      }

      const pipe = redis.pipeline();

      // Register the experiment
      pipe.sadd("tidal:experiments", expId);

      // Write metric points
      if (points && points.length > 0) {
        const serialized = points.map((p) => JSON.stringify(p));
        pipe.rpush(`tidal:metrics:${expId}:history`, ...serialized);
        pipe.ltrim(`tidal:metrics:${expId}:history`, -50000, -1);
        // Set latest to the last point in the batch
        pipe.set(
          `tidal:metrics:${expId}:latest`,
          serialized[serialized.length - 1],
          "EX",
          600,
        );
      }

      // Write status
      if (status) {
        pipe.set(`tidal:status:${expId}`, JSON.stringify(status), "EX", 900);
      }

      await pipe.exec();

      return reply.send({ ok: true, ingested: points?.length ?? 0 });
    },
  );

  // GET /api/workers/:jobId/signal — worker polls for signals
  fastify.get<{ Params: { jobId: string } }>(
    "/api/workers/:jobId/signal",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const { jobId } = request.params;
      const signal: JobSignal | null = await store.readSignal(jobId);
      if (signal) {
        await store.clearSignal(jobId);
      }
      return reply.send({ signal });
    },
  );
}
