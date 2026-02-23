import fp from "fastify-plugin";
import type { FastifyInstance } from "fastify";
import { ObjectStore } from "../services/object-store.js";

declare module "fastify" {
  interface FastifyInstance {
    objectStore: ObjectStore;
  }
}

export default fp(async function objectStorePlugin(fastify: FastifyInstance) {
  const cfg = fastify.serverConfig;

  let store: ObjectStore;

  if (
    cfg.spacesEndpoint &&
    cfg.spacesKey &&
    cfg.spacesSecret &&
    cfg.spacesBucket &&
    cfg.spacesRegion
  ) {
    store = new ObjectStore({
      endpoint: cfg.spacesEndpoint,
      region: cfg.spacesRegion,
      accessKeyId: cfg.spacesKey,
      secretAccessKey: cfg.spacesSecret,
      bucket: cfg.spacesBucket,
    });
    fastify.log.info("DigitalOcean Spaces configured");
  } else {
    store = new ObjectStore(null);
    fastify.log.info("DigitalOcean Spaces not configured — object storage disabled");
  }

  fastify.decorate("objectStore", store);
});
