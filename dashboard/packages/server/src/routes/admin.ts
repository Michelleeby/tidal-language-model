import type { FastifyInstance } from "fastify";
import type { JwtUser } from "../plugins/auth.js";

export default async function adminRoutes(fastify: FastifyInstance) {
  // -----------------------------------------------------------------------
  // GET /api/admin/allowed-users — list whitelist
  // -----------------------------------------------------------------------

  fastify.get("/api/admin/allowed-users", {
    preHandler: fastify.verifyAuth,
  }, async () => {
    return { allowedUsers: fastify.db.listAllowedUsers() };
  });

  // -----------------------------------------------------------------------
  // POST /api/admin/allowed-users — add user to whitelist
  // -----------------------------------------------------------------------

  fastify.post<{ Body: { githubLogin?: string } }>("/api/admin/allowed-users", {
    preHandler: fastify.verifyAuth,
  }, async (request, reply) => {
    const { githubLogin } = request.body ?? {};

    if (!githubLogin || !githubLogin.trim()) {
      return reply.status(400).send({ error: "githubLogin is required" });
    }

    const trimmed = githubLogin.trim();

    // Determine who is adding this user
    const addedBy = request.user?.type === "jwt"
      ? (request.user as JwtUser).githubLogin
      : null;

    const result = fastify.db.addAllowedUser(trimmed, addedBy);

    if (result) {
      return reply.status(201).send({ allowedUser: result, created: true });
    }

    // Duplicate — return the existing entry
    const existing = fastify.db.listAllowedUsers().find(
      (u) => u.githubLogin.toLowerCase() === trimmed.toLowerCase(),
    );
    return reply.status(200).send({ allowedUser: existing, created: false });
  });

  // -----------------------------------------------------------------------
  // DELETE /api/admin/allowed-users/:githubLogin — remove from whitelist
  // -----------------------------------------------------------------------

  fastify.delete<{ Params: { githubLogin: string } }>("/api/admin/allowed-users/:githubLogin", {
    preHandler: fastify.verifyAuth,
  }, async (request, reply) => {
    const { githubLogin } = request.params;

    // Block self-removal
    if (
      request.user?.type === "jwt" &&
      (request.user as JwtUser).githubLogin.toLowerCase() === githubLogin.toLowerCase()
    ) {
      return reply.status(400).send({ error: "Cannot remove yourself from the whitelist" });
    }

    const removed = fastify.db.removeAllowedUser(githubLogin);

    if (!removed) {
      return reply.status(404).send({ error: "User not found in whitelist" });
    }

    return { removed: true };
  });
}
