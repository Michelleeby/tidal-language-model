import fsp from "node:fs/promises";
import path from "node:path";
import type { FastifyInstance } from "fastify";
import type { CheckpointsResponse, CheckpointInfo } from "@tidal/shared";
import { classifyCheckpoint } from "../services/checkpoint-classifier.js";

export default async function checkpointsRoutes(fastify: FastifyInstance) {
  fastify.get<{
    Params: { expId: string };
    Reply: CheckpointsResponse;
  }>("/api/experiments/:expId/checkpoints", async (request) => {
    const { expId } = request.params;
    const expDir = path.join(fastify.serverConfig.experimentsDir, expId);

    const plugin = fastify.pluginRegistry.getDefault();
    const patterns = plugin?.checkpointPatterns ?? [];

    const checkpoints: CheckpointInfo[] = [];

    try {
      const entries = await fsp.readdir(expDir);
      for (const filename of entries) {
        if (!filename.endsWith(".pth")) continue;
        const filePath = path.join(expDir, filename);
        const stat = await fsp.stat(filePath);

        const { phase, epoch } = classifyCheckpoint(filename, patterns);

        checkpoints.push({
          filename,
          path: filePath,
          sizeBytes: stat.size,
          modified: stat.mtimeMs,
          phase,
          epoch,
        });
      }
    } catch {
      // Experiment directory may not exist
    }

    checkpoints.sort((a, b) => a.modified - b.modified);
    return { expId, checkpoints };
  });
}
