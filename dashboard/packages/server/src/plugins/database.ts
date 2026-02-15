import fp from "fastify-plugin";
import type { FastifyInstance } from "fastify";
import { Database } from "../services/database.js";

declare module "fastify" {
  interface FastifyInstance {
    db: Database;
  }
}

export default fp(async function databasePlugin(fastify: FastifyInstance) {
  const db = new Database(fastify.serverConfig.dbPath);

  fastify.decorate("db", db);

  fastify.addHook("onClose", () => {
    db.close();
  });
});
