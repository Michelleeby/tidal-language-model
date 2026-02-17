import fp from "fastify-plugin";
import type { FastifyInstance } from "fastify";
import { SSEManager, sseConfigFromManifest } from "../services/sse-manager.js";

declare module "fastify" {
  interface FastifyInstance {
    sseManager: SSEManager;
  }
}

export default fp(async function ssePlugin(fastify: FastifyInstance) {
  const plugin = fastify.tidalManifest;
  const sseConfig = plugin
    ? sseConfigFromManifest(plugin.redis)
    : undefined;

  const manager = new SSEManager(fastify.redis, undefined, sseConfig);

  fastify.decorate("sseManager", manager);

  fastify.addHook("onClose", () => manager.stop());
});
