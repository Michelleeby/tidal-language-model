import fsp from "node:fs/promises";
import path from "node:path";
import type { FastifyInstance } from "fastify";
import type { EvaluationResponse, AblationResponse } from "@tidal/shared";

export default async function evaluationRoutes(fastify: FastifyInstance) {
  fastify.get<{
    Params: { expId: string };
    Reply: EvaluationResponse;
  }>("/api/experiments/:expId/evaluation", async (request) => {
    const { expId } = request.params;
    const filePath = path.join(
      fastify.serverConfig.experimentsDir,
      expId,
      "evaluation_results.json",
    );
    try {
      const content = await fsp.readFile(filePath, "utf-8");
      return { expId, results: JSON.parse(content) };
    } catch {
      return { expId, results: null };
    }
  });

  fastify.get<{
    Params: { expId: string };
    Reply: AblationResponse;
  }>("/api/experiments/:expId/ablation", async (request) => {
    const { expId } = request.params;
    const filePath = path.join(
      fastify.serverConfig.experimentsDir,
      expId,
      "ablation_results.json",
    );
    try {
      const content = await fsp.readFile(filePath, "utf-8");
      return { expId, results: JSON.parse(content) };
    } catch {
      return { expId, results: null };
    }
  });
}
