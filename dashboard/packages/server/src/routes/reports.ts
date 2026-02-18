import type { FastifyInstance } from "fastify";
import type { CreateReportRequest, UpdateReportRequest, GenerateReportRequest } from "@tidal/shared";
import { buildPatternBlocks } from "@tidal/shared";

export default async function reportsRoutes(fastify: FastifyInstance) {
  // List all reports (summaries only)
  fastify.get("/api/reports", async () => {
    const reports = fastify.db.listReports();
    return { reports };
  });

  // Get a single report by id
  fastify.get<{ Params: { id: string } }>(
    "/api/reports/:id",
    async (request, reply) => {
      const report = fastify.db.getReport(request.params.id);
      if (!report) {
        return reply.status(404).send({ error: "Report not found" });
      }
      return { report };
    },
  );

  // Create a new report
  fastify.post<{ Body: CreateReportRequest }>(
    "/api/reports",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const userId =
        request.user?.type === "jwt" ? request.user.userId : null;
      const report = fastify.db.createReport(request.body?.title, userId ?? undefined);
      return reply.status(201).send({ report });
    },
  );

  // Update an existing report
  fastify.put<{ Params: { id: string }; Body: UpdateReportRequest }>(
    "/api/reports/:id",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const report = fastify.db.updateReport(request.params.id, {
        title: request.body?.title,
        blocks: request.body?.blocks,
      });
      if (!report) {
        return reply.status(404).send({ error: "Report not found" });
      }
      return { report };
    },
  );

  // Generate a report from a block pattern
  fastify.post<{ Body: GenerateReportRequest }>(
    "/api/reports/generate",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const { pattern, experimentId, title, githubLogin } = request.body ?? {};

      if (!experimentId) {
        return reply.status(400).send({ error: "experimentId is required" });
      }
      if (!pattern) {
        return reply.status(400).send({ error: "pattern is required" });
      }

      const blocks = buildPatternBlocks(pattern, experimentId);
      if (!blocks) {
        return reply.status(400).send({ error: `Unknown pattern: ${pattern}` });
      }

      // Resolve user from githubLogin if provided
      let userId: string | undefined;
      if (githubLogin) {
        const user = fastify.db.getUserByGithubLogin(githubLogin);
        if (user) userId = user.id;
      }

      const report = fastify.db.createReport(
        title ?? `${pattern} — ${experimentId}`,
        userId,
      );
      const updated = fastify.db.updateReport(report.id, { blocks });
      if (!updated) {
        return reply.status(500).send({ error: "Failed to persist report blocks" });
      }

      return reply.status(201).send({ report: updated });
    },
  );

  // Delete a report
  fastify.delete<{ Params: { id: string } }>(
    "/api/reports/:id",
    { preHandler: [fastify.verifyAuth] },
    async (request, reply) => {
      const deleted = fastify.db.deleteReport(request.params.id);
      if (!deleted) {
        return reply.status(404).send({ error: "Report not found" });
      }
      return { deleted: true };
    },
  );
}
