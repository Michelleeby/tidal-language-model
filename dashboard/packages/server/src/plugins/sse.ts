import fp from "fastify-plugin";
import type { FastifyInstance } from "fastify";
import { SSEManager } from "../services/sse-manager.js";

declare module "fastify" {
  interface FastifyInstance {
    sseManager: SSEManager;
  }
}

export default fp(async function ssePlugin(fastify: FastifyInstance) {
  const manager = new SSEManager(fastify.redis);

  fastify.decorate("sseManager", manager);

  fastify.addHook("onClose", () => manager.stop());
});
