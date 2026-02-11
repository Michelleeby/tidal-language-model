import type { FastifyInstance } from "fastify";

export default async function sseRoutes(fastify: FastifyInstance) {
  const manager = fastify.sseManager;

  fastify.get<{
    Params: { expId: string };
  }>("/api/experiments/:expId/stream", async (request, reply) => {
    reply.raw.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });

    manager.addClient(request.params.expId, reply);

    // Keep the connection open — Fastify will not auto-close raw streams
    // The SSEManager handles cleanup when the client disconnects
    await new Promise(() => {
      // Never resolves — held open until client disconnects
    });
  });

  fastify.get("/api/jobs/stream", async (_request, reply) => {
    reply.raw.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });

    manager.addGlobalClient(reply);

    await new Promise(() => {
      // Never resolves — held open until client disconnects
    });
  });
}
