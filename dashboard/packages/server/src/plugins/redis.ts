import fp from "fastify-plugin";
import Redis from "ioredis";
import type { FastifyInstance } from "fastify";

declare module "fastify" {
  interface FastifyInstance {
    redis: Redis | null;
  }
}

export default fp(async function redisPlugin(
  fastify: FastifyInstance,
  opts: { url: string },
) {
  let redis: Redis | null = null;

  try {
    redis = new Redis(opts.url, {
      maxRetriesPerRequest: 1,
      retryStrategy: (times) => (times > 2 ? null : Math.min(times * 200, 1000)),
      lazyConnect: true,
    });
    await redis.connect();
    fastify.log.info("Redis connected");
  } catch {
    fastify.log.warn("Redis unavailable — running in disk-only mode");
    redis = null;
  }

  fastify.decorate("redis", redis);

  fastify.addHook("onClose", async () => {
    if (redis) {
      await redis.quit();
    }
  });
});
