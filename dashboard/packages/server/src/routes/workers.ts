import type { FastifyInstance } from "fastify";
import type { JobSignal, JobStatus } from "@tidal/shared";
import * as fsp from "node:fs/promises";
import * as path from "node:path";
import { pipeline } from "node:stream/promises";
import { createWriteStream } from "node:fs";
import type { Readable } from "node:stream";
import { JobStore } from "../services/job-store.js";
import type { SSEManager } from "../services/sse-manager.js";
import { ExperimentArchiver } from "../services/experiment-archiver.js";

const CHECKPOINT_FILENAME_RE = /^[\w.-]+\.pth$/;

export default async function workerRoutes(fastify: FastifyInstance) {
  const store = new JobStore(fastify.redis);
  const sseManager: SSEManager = fastify.sseManager;
  const experimentsDir = fastify.serverConfig.experimentsDir;
  const archiver = new ExperimentArchiver(
    fastify.redis,
    experimentsDir,
    fastify.log,
  );

  // Register raw body parser for checkpoint uploads — pass stream through as-is
  fastify.addContentTypeParser(
    "application/octet-stream",
    (_req: unknown, payload: Readable, done: (err: Error | null, body?: Readable) => void) => {
      done(null, payload);
    },
  );

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

      // Archive experiment data to disk before Redis TTLs expire
      if (status === "completed" || status === "failed") {
        const expId = updated.experimentId;
        if (expId) {
          try {
            await archiver.archive(expId);
          } catch (err) {
            fastify.log.error({ expId, err }, "Archival failed");
          }
        }
      }

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

  // PATCH /api/workers/:jobId/experiment-id — worker reports its experiment ID
  fastify.patch<{
    Params: { jobId: string };
    Body: { experimentId: string };
  }>(
    "/api/workers/:jobId/experiment-id",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const { jobId } = request.params;
      const { experimentId } = request.body;

      if (!experimentId) {
        return reply.status(400).send({ error: "experimentId is required" });
      }

      const updated = await store.update(jobId, { experimentId });
      if (!updated) {
        return reply.status(404).send({ error: "Job not found" });
      }

      sseManager.broadcastJobUpdate(updated);
      return reply.send({ ok: true, experimentId });
    },
  );

  // PUT /api/workers/:jobId/checkpoints/:filename — worker uploads a checkpoint file
  // Supports chunked uploads via ?chunk=N&totalChunks=M query params to stay
  // under Cloudflare's 100MB body size limit.
  fastify.put<{
    Params: { jobId: string; filename: string };
    Querystring: { expId?: string; chunk?: string; totalChunks?: string };
  }>(
    "/api/workers/:jobId/checkpoints/:filename",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const { jobId, filename } = request.params;

      // Validate filename to prevent path traversal
      if (!CHECKPOINT_FILENAME_RE.test(filename)) {
        return reply
          .status(400)
          .send({ error: "Invalid filename — must match /^[\\w.-]+\\.pth$/" });
      }

      // Resolve experiment ID from job or query param fallback
      const job = await store.get(jobId);
      if (!job) {
        return reply.status(404).send({ error: "Job not found" });
      }

      const expId = job.experimentId ?? request.query.expId;
      if (!expId) {
        return reply.status(400).send({
          error:
            "No experimentId on job — pass ?expId= query param as fallback",
        });
      }

      const expDir = path.join(experimentsDir, expId);
      await fsp.mkdir(expDir, { recursive: true });

      const dest = path.join(expDir, filename);
      const chunkParam = request.query.chunk;
      const totalChunksParam = request.query.totalChunks;
      const isChunked =
        chunkParam !== undefined && totalChunksParam !== undefined;

      if (!isChunked) {
        // Single-file upload (small checkpoints or non-Cloudflare paths)
        const tmpDest = dest + ".tmp";
        try {
          const bodyStream = request.body as Readable;
          await pipeline(bodyStream, createWriteStream(tmpDest));
          await fsp.rename(tmpDest, dest);

          const stat = await fsp.stat(dest);
          fastify.log.info(
            { jobId, expId, filename, sizeBytes: stat.size },
            "Checkpoint uploaded",
          );
          return reply.send({ ok: true, filename, sizeBytes: stat.size });
        } catch (err) {
          await fsp.unlink(tmpDest).catch(() => {});
          throw err;
        }
      }

      // Chunked upload — write part file, assemble when all parts arrive
      const chunkIdx = parseInt(chunkParam, 10);
      const totalChunks = parseInt(totalChunksParam, 10);
      if (
        isNaN(chunkIdx) ||
        isNaN(totalChunks) ||
        chunkIdx < 0 ||
        chunkIdx >= totalChunks ||
        totalChunks < 1
      ) {
        return reply
          .status(400)
          .send({ error: "Invalid chunk/totalChunks params" });
      }

      const partFile = `${dest}.part.${chunkIdx}`;
      const tmpPart = partFile + ".tmp";
      try {
        const bodyStream = request.body as Readable;
        await pipeline(bodyStream, createWriteStream(tmpPart));
        await fsp.rename(tmpPart, partFile);
      } catch (err) {
        await fsp.unlink(tmpPart).catch(() => {});
        throw err;
      }

      // Check if all parts have arrived
      const partsPresent = await Promise.all(
        Array.from({ length: totalChunks }, (_, i) =>
          fsp
            .access(`${dest}.part.${i}`)
            .then(() => true)
            .catch(() => false),
        ),
      );

      if (!partsPresent.every(Boolean)) {
        fastify.log.info(
          { jobId, expId, filename, chunkIdx, totalChunks },
          "Chunk received, waiting for remaining parts",
        );
        return reply.send({
          ok: true,
          filename,
          chunk: chunkIdx,
          totalChunks,
          assembled: false,
        });
      }

      // All parts present — concatenate into final file
      const tmpDest = dest + ".assembling";
      try {
        const ws = createWriteStream(tmpDest);
        for (let i = 0; i < totalChunks; i++) {
          const { createReadStream } = await import("node:fs");
          const rs = createReadStream(`${dest}.part.${i}`);
          await pipeline(rs, ws, { end: false });
        }
        ws.end();
        await new Promise<void>((resolve, reject) => {
          ws.on("finish", resolve);
          ws.on("error", reject);
        });

        await fsp.rename(tmpDest, dest);

        // Clean up part files
        await Promise.all(
          Array.from({ length: totalChunks }, (_, i) =>
            fsp.unlink(`${dest}.part.${i}`).catch(() => {}),
          ),
        );

        const stat = await fsp.stat(dest);
        fastify.log.info(
          { jobId, expId, filename, sizeBytes: stat.size, totalChunks },
          "Chunked checkpoint assembled",
        );
        return reply.send({
          ok: true,
          filename,
          sizeBytes: stat.size,
          assembled: true,
        });
      } catch (err) {
        await fsp.unlink(tmpDest).catch(() => {});
        throw err;
      }
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
