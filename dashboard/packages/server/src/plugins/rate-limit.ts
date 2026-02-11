import fp from "fastify-plugin";
import type { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";

declare module "fastify" {
  interface FastifyInstance {
    rateLimit: (
      request: FastifyRequest,
      reply: FastifyReply,
    ) => Promise<void>;
  }
}

const CAPACITY = 5;
const REFILL_INTERVAL_MS = 6000; // 1 token per 6 seconds
const KEY_TTL_S = 120;

// Atomic token bucket via Lua: refills based on elapsed time, then consumes.
// Returns [allowed (0/1), remaining tokens, ms until next token].
const TOKEN_BUCKET_LUA = `
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_ms = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

local tokens = tonumber(redis.call('HGET', key, 't') or capacity)
local last = tonumber(redis.call('HGET', key, 'ts') or now)

local elapsed = math.max(now - last, 0)
local refill = math.floor(elapsed / refill_ms)
tokens = math.min(capacity, tokens + refill)
local new_last = last + refill * refill_ms

if tokens > 0 then
  tokens = tokens - 1
  redis.call('HMSET', key, 't', tokens, 'ts', new_last)
  redis.call('EXPIRE', key, ttl)
  return {1, tokens, 0}
else
  local wait = refill_ms - (now - new_last)
  return {0, 0, wait}
end
`;

export default fp(async function rateLimitPlugin(fastify: FastifyInstance) {
  async function rateLimit(
    request: FastifyRequest,
    reply: FastifyReply,
  ): Promise<void> {
    const redis = fastify.redis;
    if (!redis) return; // fail-open when Redis is down

    const ip = request.ip;
    const key = `rl:gen:${ip}`;
    const now = Date.now();

    try {
      const result = (await redis.eval(
        TOKEN_BUCKET_LUA,
        1,
        key,
        CAPACITY,
        REFILL_INTERVAL_MS,
        now,
        KEY_TTL_S,
      )) as [number, number, number];

      const [allowed, remaining, waitMs] = result;

      reply.header("X-RateLimit-Limit", CAPACITY);
      reply.header("X-RateLimit-Remaining", remaining);

      if (!allowed) {
        const retryAfter = Math.ceil(waitMs / 1000);
        reply.header("Retry-After", retryAfter);
        reply.status(429).send({
          error: `Rate limited. Try again in ${retryAfter} seconds.`,
        });
      }
    } catch (err) {
      // fail-open: allow request if Redis errors
      fastify.log.warn({ err }, "Rate limit Redis error — allowing request");
    }
  }

  fastify.decorate("rateLimit", rateLimit);
});
