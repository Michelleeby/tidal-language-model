import type { FastifyInstance } from "fastify";
import type { CreateAnalysisRequest, AnalysisType } from "@tidal/shared";

const VALID_TYPES: ReadonlySet<string> = new Set([
  "trajectory",
  "cross-prompt",
  "sweep",
]);

export default async function analysesRoutes(fastify: FastifyInstance) {
  // List analyses for an experiment (summaries only, no data blob)
  fastify.get<{
    Params: { expId: string };
    Querystring: { type?: string };
  }>("/api/experiments/:expId/analyses", async (request) => {
    let analyses = fastify.db.listAnalyses(request.params.expId);

    const typeFilter = request.query.type;
    if (typeFilter && VALID_TYPES.has(typeFilter)) {
      analyses = analyses.filter((a) => a.analysisType === typeFilter);
    }

    return { analyses };
  });

  // Get a full analysis result (includes data blob)
  fastify.get<{ Params: { id: string } }>(
    "/api/analyses/:id",
    async (request, reply) => {
      const analysis = fastify.db.getAnalysis(request.params.id);
      if (!analysis) {
        return reply.status(404).send({ error: "Analysis not found" });
      }
      return { analysis };
    },
  );

  // Create a new analysis result
  fastify.post<{ Params: { expId: string }; Body: CreateAnalysisRequest }>(
    "/api/experiments/:expId/analyses",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const { analysisType, label, request: reqData, data } =
        request.body ?? ({} as Partial<CreateAnalysisRequest>);

      if (!analysisType || !VALID_TYPES.has(analysisType)) {
        return reply
          .status(400)
          .send({ error: "analysisType is required (trajectory, cross-prompt, sweep)" });
      }
      if (!data || typeof data !== "object") {
        return reply.status(400).send({ error: "data is required" });
      }

      const analysis = fastify.db.createAnalysis({
        experimentId: request.params.expId,
        analysisType: analysisType as AnalysisType,
        label: label || `${analysisType} analysis`,
        request: reqData ?? {},
        data,
      });

      return reply.status(201).send({ analysis });
    },
  );

  // Delete an analysis result
  fastify.delete<{ Params: { id: string } }>(
    "/api/analyses/:id",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const deleted = fastify.db.deleteAnalysis(request.params.id);
      if (!deleted) {
        return reply.status(404).send({ error: "Analysis not found" });
      }
      return { deleted: true };
    },
  );
}
