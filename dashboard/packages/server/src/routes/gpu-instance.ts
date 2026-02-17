import fsp from "node:fs/promises";
import path from "node:path";
import type { FastifyInstance } from "fastify";
import type { GpuInstanceResponse } from "@tidal/shared";

export default async function gpuInstanceRoutes(fastify: FastifyInstance) {
  fastify.get<{
    Params: { expId: string };
    Reply: GpuInstanceResponse;
  }>("/api/experiments/:expId/gpu-instance", async (request) => {
    const { expId } = request.params;
    const filePath = path.join(
      fastify.serverConfig.experimentsDir,
      expId,
      "gpu_instance.json",
    );
    try {
      const content = await fsp.readFile(filePath, "utf-8");
      return { expId, instance: JSON.parse(content) };
    } catch {
      return { expId, instance: null };
    }
  });
}
