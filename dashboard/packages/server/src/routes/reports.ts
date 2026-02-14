import type { FastifyInstance } from "fastify";
import path from "node:path";
import { ReportStore } from "../services/report-store.js";
import type { CreateReportRequest, UpdateReportRequest } from "@tidal/shared";

export default async function reportsRoutes(fastify: FastifyInstance) {
  const reportsDir = path.join(fastify.serverConfig.projectRoot, "reports");
  const store = new ReportStore(reportsDir);

  // List all reports (summaries only)
  fastify.get("/api/reports", async () => {
    const reports = await store.list();
    return { reports };
  });

  // Get a single report by id
  fastify.get<{ Params: { id: string } }>(
    "/api/reports/:id",
    async (request, reply) => {
      const report = await store.get(request.params.id);
      if (!report) {
        return reply.status(404).send({ error: "Report not found" });
      }
      return { report };
    },
  );

  // Create a new report
  fastify.post<{ Body: CreateReportRequest }>(
    "/api/reports",
    async (request, reply) => {
      const report = await store.create(request.body?.title);
      return reply.status(201).send({ report });
    },
  );

  // Update an existing report
  fastify.put<{ Params: { id: string }; Body: UpdateReportRequest }>(
    "/api/reports/:id",
    async (request, reply) => {
      const report = await store.update(request.params.id, {
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
    async (request, reply) => {
      const deleted = await store.delete(request.params.id);
      if (!deleted) {
        return reply.status(404).send({ error: "Report not found" });
      }
      return { deleted: true };
    },
  );
}
