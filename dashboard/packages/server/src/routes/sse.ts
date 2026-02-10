import type { FastifyInstance } from "fastify";
import { SSEManager } from "../services/sse-manager.js";

export default async function sseRoutes(fastify: FastifyInstance) {
  const manager = new SSEManager(fastify.redis);

  fastify.addHook("onClose", () => manager.stop());

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
}
