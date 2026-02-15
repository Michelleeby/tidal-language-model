import type { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";
import type {
  CreateUserPluginRequest,
  UpdateUserPluginRequest,
  PluginFileWriteRequest,
  UserPlugin,
} from "@tidal/shared";

const PLUGIN_NAME_RE = /^[a-z][a-z0-9_-]{1,48}[a-z0-9]$/;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function requireJwtUser(
  request: FastifyRequest,
  reply: FastifyReply,
): string | null {
  if (request.user?.type !== "jwt") {
    reply.status(403).send({ error: "JWT authentication required" });
    return null;
  }
  return request.user.userId;
}

async function loadOwnedPlugin(
  fastify: FastifyInstance,
  pluginId: string,
  userId: string,
  reply: FastifyReply,
): Promise<UserPlugin | null> {
  const plugin = fastify.db.getUserPlugin(pluginId);
  if (!plugin || plugin.userId !== userId) {
    reply.status(404).send({ error: "Plugin not found" });
    return null;
  }
  return plugin;
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

export default async function userPluginsRoutes(fastify: FastifyInstance) {
  // All routes require auth
  const preHandler = [fastify.verifyAuth];

  // -------------------------------------------------------------------------
  // Plugin CRUD
  // -------------------------------------------------------------------------

  /** GET /api/user-plugins — list current user's plugins. */
  fastify.get(
    "/api/user-plugins",
    { preHandler },
    async (request, reply) => {
      const userId = requireJwtUser(request, reply);
      if (!userId) return;

      const plugins = fastify.db.listUserPluginsByUser(userId);
      return { plugins };
    },
  );

  /** POST /api/user-plugins — create plugin from template. */
  fastify.post<{ Body: CreateUserPluginRequest }>(
    "/api/user-plugins",
    { preHandler },
    async (request, reply) => {
      const userId = requireJwtUser(request, reply);
      if (!userId) return;

      const { name, displayName } = request.body;

      // Validate name format
      if (!PLUGIN_NAME_RE.test(name)) {
        return reply.status(400).send({
          error:
            "Invalid plugin name. Must match /^[a-z][a-z0-9_-]{1,48}[a-z0-9]$/",
        });
      }

      // Check collision with system plugins
      if (fastify.pluginRegistry.get(name)) {
        return reply.status(409).send({
          error: `Name "${name}" conflicts with a system plugin`,
        });
      }

      // Check duplicate for this user
      if (fastify.db.getUserPluginByUserAndName(userId, name)) {
        return reply.status(409).send({
          error: `You already have a plugin named "${name}"`,
        });
      }

      // Create DB record + copy template files
      const plugin = fastify.db.createUserPlugin({
        userId,
        name,
        displayName,
      });

      await fastify.userPluginStore.createFromTemplate(userId, name);

      return reply.status(201).send({ plugin });
    },
  );

  /** GET /api/user-plugins/:id — get plugin metadata. */
  fastify.get<{ Params: { id: string } }>(
    "/api/user-plugins/:id",
    { preHandler },
    async (request, reply) => {
      const userId = requireJwtUser(request, reply);
      if (!userId) return;

      const plugin = await loadOwnedPlugin(
        fastify,
        request.params.id,
        userId,
        reply,
      );
      if (!plugin) return;

      return { plugin };
    },
  );

  /** PUT /api/user-plugins/:id — update display name. */
  fastify.put<{ Params: { id: string }; Body: UpdateUserPluginRequest }>(
    "/api/user-plugins/:id",
    { preHandler },
    async (request, reply) => {
      const userId = requireJwtUser(request, reply);
      if (!userId) return;

      const existing = await loadOwnedPlugin(
        fastify,
        request.params.id,
        userId,
        reply,
      );
      if (!existing) return;

      const plugin = fastify.db.updateUserPlugin(request.params.id, {
        displayName: request.body.displayName,
      });
      return { plugin };
    },
  );

  /** DELETE /api/user-plugins/:id — delete plugin + files. */
  fastify.delete<{ Params: { id: string } }>(
    "/api/user-plugins/:id",
    { preHandler },
    async (request, reply) => {
      const userId = requireJwtUser(request, reply);
      if (!userId) return;

      const plugin = await loadOwnedPlugin(
        fastify,
        request.params.id,
        userId,
        reply,
      );
      if (!plugin) return;

      // Delete files first, then DB record
      await fastify.userPluginStore.deletePlugin(userId, plugin.name);
      fastify.db.deleteUserPlugin(request.params.id);

      return { deleted: true };
    },
  );

  // -------------------------------------------------------------------------
  // File operations
  // -------------------------------------------------------------------------

  /** GET /api/user-plugins/:id/files — recursive file tree. */
  fastify.get<{ Params: { id: string } }>(
    "/api/user-plugins/:id/files",
    { preHandler },
    async (request, reply) => {
      const userId = requireJwtUser(request, reply);
      if (!userId) return;

      const plugin = await loadOwnedPlugin(
        fastify,
        request.params.id,
        userId,
        reply,
      );
      if (!plugin) return;

      const files = await fastify.userPluginStore.getFileTree(
        userId,
        plugin.name,
      );
      return { files };
    },
  );

  /** GET /api/user-plugins/:id/files/* — read file content. */
  fastify.get<{ Params: { id: string; "*": string } }>(
    "/api/user-plugins/:id/files/*",
    { preHandler },
    async (request, reply) => {
      const userId = requireJwtUser(request, reply);
      if (!userId) return;

      const plugin = await loadOwnedPlugin(
        fastify,
        request.params.id,
        userId,
        reply,
      );
      if (!plugin) return;

      const relativePath = request.params["*"];

      try {
        const content = await fastify.userPluginStore.readFile(
          userId,
          plugin.name,
          relativePath,
        );
        return { path: relativePath, content };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.toLowerCase().includes("path")) {
          return reply.status(400).send({ error: msg });
        }
        return reply.status(404).send({ error: "File not found" });
      }
    },
  );

  /** PUT /api/user-plugins/:id/files/* — save file content. */
  fastify.put<{ Params: { id: string; "*": string }; Body: PluginFileWriteRequest }>(
    "/api/user-plugins/:id/files/*",
    { preHandler },
    async (request, reply) => {
      const userId = requireJwtUser(request, reply);
      if (!userId) return;

      const plugin = await loadOwnedPlugin(
        fastify,
        request.params.id,
        userId,
        reply,
      );
      if (!plugin) return;

      const relativePath = request.params["*"];

      try {
        await fastify.userPluginStore.writeFile(
          userId,
          plugin.name,
          relativePath,
          request.body.content,
        );
        return { path: relativePath, saved: true };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        return reply.status(400).send({ error: msg });
      }
    },
  );

  /** POST /api/user-plugins/:id/files/* — create new file. */
  fastify.post<{ Params: { id: string; "*": string }; Body: PluginFileWriteRequest }>(
    "/api/user-plugins/:id/files/*",
    { preHandler },
    async (request, reply) => {
      const userId = requireJwtUser(request, reply);
      if (!userId) return;

      const plugin = await loadOwnedPlugin(
        fastify,
        request.params.id,
        userId,
        reply,
      );
      if (!plugin) return;

      const relativePath = request.params["*"];

      try {
        await fastify.userPluginStore.writeFile(
          userId,
          plugin.name,
          relativePath,
          request.body.content,
        );
        return reply.status(201).send({ path: relativePath, created: true });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        return reply.status(400).send({ error: msg });
      }
    },
  );

  /** DELETE /api/user-plugins/:id/files/* — delete a file. */
  fastify.delete<{ Params: { id: string; "*": string } }>(
    "/api/user-plugins/:id/files/*",
    { preHandler },
    async (request, reply) => {
      const userId = requireJwtUser(request, reply);
      if (!userId) return;

      const plugin = await loadOwnedPlugin(
        fastify,
        request.params.id,
        userId,
        reply,
      );
      if (!plugin) return;

      const relativePath = request.params["*"];

      try {
        await fastify.userPluginStore.deleteFile(
          userId,
          plugin.name,
          relativePath,
        );
        return { path: relativePath, deleted: true };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.toLowerCase().includes("manifest")) {
          return reply.status(400).send({ error: msg });
        }
        return reply.status(404).send({ error: "File not found" });
      }
    },
  );
}
