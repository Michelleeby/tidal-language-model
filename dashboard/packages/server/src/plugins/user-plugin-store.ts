import fp from "fastify-plugin";
import path from "node:path";
import type { FastifyInstance } from "fastify";
import { UserPluginStore } from "../services/user-plugin-store.js";

declare module "fastify" {
  interface FastifyInstance {
    userPluginStore: UserPluginStore;
  }
}

export default fp(async function userPluginStorePlugin(
  fastify: FastifyInstance,
) {
  const { userPluginsDir, projectRoot } = fastify.serverConfig;
  const templateDir = path.join(projectRoot, "plugins", "tidal");

  const store = new UserPluginStore(userPluginsDir, templateDir);
  fastify.decorate("userPluginStore", store);
});
