import type { FastifyInstance } from "fastify";
import type { CreateReportRequest, UpdateReportRequest } from "@tidal/shared";

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
