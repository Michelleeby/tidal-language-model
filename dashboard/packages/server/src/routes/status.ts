import type { FastifyInstance } from "fastify";
import { MetricsReader, metricsReaderConfigFromManifest } from "../services/metrics-reader.js";
import { JobStore, jobStoreKeysFromManifest } from "../services/job-store.js";
import type { StatusResponse, MarkCompleteResponse } from "@tidal/shared";

const STALENESS_THRESHOLD_S = 300; // 5 minutes

export default async function statusRoutes(fastify: FastifyInstance) {
  const plugin = fastify.tidalManifest;
  const metricsConfig = plugin
    ? metricsReaderConfigFromManifest(plugin.metrics)
    : undefined;

  const reader = new MetricsReader(
    fastify.redis,
    fastify.serverConfig.experimentsDir,
    metricsConfig,
  );

  const storeKeys = plugin
    ? jobStoreKeysFromManifest(plugin.redis)
    : undefined;
  const jobStore = new JobStore(fastify.redis, undefined, storeKeys);

  fastify.get<{
    Params: { expId: string };
    Reply: StatusResponse;
  }>("/api/experiments/:expId/status", async (request) => {
    const { expId } = request.params;
    const status = await reader.getStatus(expId);
    return { expId, status };
  });

  fastify.post<{
    Params: { expId: string };
    Reply: MarkCompleteResponse;
  }>("/api/experiments/:expId/status/complete", {
    preHandler: [fastify.verifyAuth],
  }, async (request, reply) => {
    const { expId } = request.params;

    const status = await reader.getStatus(expId);
    if (!status) {
      return reply.status(404).send({ error: "No status found for experiment" } as any);
    }

    // Idempotent: already completed
    if (status.status === "completed") {
      return { expId, status };
    }

    // Safety guard: staleness check
    if (status.last_update) {
      const ageS = Date.now() / 1000 - status.last_update;
      if (ageS < STALENESS_THRESHOLD_S) {
        return reply.status(409).send({
          error: `Experiment appears to still be active (last updated ${Math.round(ageS)}s ago). Wait for training to finish or stop the job first.`,
        } as any);
      }
    }

    // Safety guard: active job check
    try {
      const activeJobs = await jobStore.listActive();
      const linkedJob = activeJobs.find((j) => j.experimentId === expId);
      if (linkedJob) {
        return reply.status(409).send({
          error: `Experiment has an active job (${linkedJob.jobId}). Stop the job before marking complete.`,
        } as any);
      }
    } catch {
      // Redis unavailable — skip job check
    }

    // Write completed status
    const updatedStatus = {
      ...status,
      status: "completed" as const,
      end_time: Date.now() / 1000,
    };
    await reader.writeStatus(expId, updatedStatus);

    return { expId, status: updatedStatus };
  });
}
