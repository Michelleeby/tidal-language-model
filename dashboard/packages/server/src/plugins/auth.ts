import fp from "fastify-plugin";
import { timingSafeEqual } from "node:crypto";
import { jwtVerify } from "jose";
import type { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";

// ---------------------------------------------------------------------------
// Type augmentation — request.user is set by verifyAuth
// ---------------------------------------------------------------------------

export interface BearerUser {
  type: "bearer";
}

export interface JwtUser {
  type: "jwt";
  userId: string;
  githubLogin: string;
}

export type AuthUser = BearerUser | JwtUser;

declare module "fastify" {
  interface FastifyRequest {
    user?: AuthUser;
  }
  interface FastifyInstance {
    verifyAuth: (
      request: FastifyRequest,
      reply: FastifyReply,
    ) => Promise<void>;
  }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

export default fp(async function authPlugin(fastify: FastifyInstance) {
  const configToken = fastify.serverConfig.authToken;
  const jwtSecret = fastify.serverConfig.jwtSecret;

  // Encode JWT secret once for jose
  const jwtKey = jwtSecret ? new TextEncoder().encode(jwtSecret) : null;

  async function verifyAuth(
    request: FastifyRequest,
    reply: FastifyReply,
  ): Promise<void> {
    // 0. Dev mode bypass — auto-authenticate as synthetic user
    if (fastify.serverConfig.devMode) {
      request.user = { type: "jwt", userId: "dev", githubLogin: "dev" };
      return;
    }

    // 1. Try JWT cookie first
    if (jwtKey) {
      const cookieToken =
        request.cookies?.["tidal_session"] as string | undefined;
      if (cookieToken) {
        try {
          const { payload } = await jwtVerify(cookieToken, jwtKey);
          request.user = {
            type: "jwt",
            userId: payload.sub as string,
            githubLogin: payload.githubLogin as string,
          };
          return;
        } catch {
          // Invalid/expired JWT — fall through to Bearer check
        }
      }
    }

    // 2. Try Bearer token
    if (configToken) {
      const header = request.headers.authorization;
      if (header?.startsWith("Bearer ")) {
        const provided = header.slice(7);
        const a = Buffer.from(provided, "utf-8");
        const b = Buffer.from(configToken, "utf-8");

        if (a.length === b.length && timingSafeEqual(a, b)) {
          request.user = { type: "bearer" };
          return;
        }
      }
    }

    // 3. No valid credentials
    reply
      .status(401)
      .send({ error: "Missing or invalid authentication credentials" });
  }

  fastify.decorate("verifyAuth", verifyAuth);
});
