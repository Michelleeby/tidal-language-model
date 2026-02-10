import fsp from "node:fs/promises";
import path from "node:path";
import type { FastifyInstance } from "fastify";
import type { CheckpointsResponse, CheckpointInfo } from "@tidal/shared";

export default async function checkpointsRoutes(fastify: FastifyInstance) {
  fastify.get<{
    Params: { expId: string };
    Reply: CheckpointsResponse;
  }>("/api/experiments/:expId/checkpoints", async (request) => {
    const { expId } = request.params;
    const expDir = path.join(fastify.serverConfig.experimentsDir, expId);

    const checkpoints: CheckpointInfo[] = [];

    try {
      const entries = await fsp.readdir(expDir);
      for (const filename of entries) {
        if (!filename.endsWith(".pth")) continue;
        const filePath = path.join(expDir, filename);
        const stat = await fsp.stat(filePath);

        // Parse phase and epoch from filename
        let phase = "unknown";
        let epoch: number | undefined;

        if (filename.startsWith("checkpoint_foundational")) {
          phase = "foundational";
          const match = filename.match(/epoch_(\d+)/);
          if (match) epoch = parseInt(match[1], 10);
        } else if (filename.startsWith("rl_checkpoint")) {
          phase = "rl";
          const match = filename.match(/iter_(\d+)/);
          if (match) epoch = parseInt(match[1], 10);
        } else if (filename.includes("_v")) {
          phase = "final";
        }

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
