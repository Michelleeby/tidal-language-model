import fp from "fastify-plugin";
import { timingSafeEqual } from "node:crypto";
import type { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";

declare module "fastify" {
  interface FastifyInstance {
    verifyAuth: (
      request: FastifyRequest,
      reply: FastifyReply,
    ) => Promise<void>;
  }
}

export default fp(async function authPlugin(fastify: FastifyInstance) {
  const configToken = fastify.serverConfig.authToken;

  if (configToken === null) {
    throw new Error(
      "TIDAL_AUTH_TOKEN environment variable is required. Set it before starting the server.",
    );
  }

  const authToken: string = configToken;

  async function verifyAuth(
    request: FastifyRequest,
    reply: FastifyReply,
  ): Promise<void> {
    const header = request.headers.authorization;
    if (!header || !header.startsWith("Bearer ")) {
      reply
        .status(401)
        .send({ error: "Missing or malformed Authorization header" });
      return;
    }

    const provided = header.slice(7);
    const expected = authToken;

    const a = Buffer.from(provided, "utf-8");
    const b = Buffer.from(expected, "utf-8");

    if (a.length !== b.length || !timingSafeEqual(a, b)) {
      reply.status(401).send({ error: "Invalid token" });
      return;
    }
  }

  fastify.decorate("verifyAuth", verifyAuth);
});
