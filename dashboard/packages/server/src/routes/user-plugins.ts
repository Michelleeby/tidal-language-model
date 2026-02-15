import fsp from "node:fs/promises";
import path from "node:path";
import { parse as parseYaml } from "yaml";
import type { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";
import type {
  CreateUserPluginRequest,
  UpdateUserPluginRequest,
  PluginFileWriteRequest,
  PluginGitPushRequest,
  UserPlugin,
} from "@tidal/shared";

const PLUGIN_NAME_RE = /^[a-z][a-z0-9_]{1,48}[a-z0-9]$/;

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
            "Invalid plugin name. Use lowercase letters, digits, and underscores only. Must match /^[a-z][a-z0-9_]{1,48}[a-z0-9]$/",
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

      // Check if user has a GitHub token for repo creation
      const user = fastify.db.getUserById(userId);
      const ghToken = user?.githubAccessToken;
      let githubRepoUrl = "";

      if (ghToken) {
        // Create GitHub repo (reuses existing on 422)
        const repo = await fastify.githubRepo.createRepo(
          ghToken,
          name,
          `Tidal plugin: ${displayName}`,
        );
        githubRepoUrl = repo.htmlUrl;

        // Clone repo locally; if dir already exists, pull instead
        const pluginDir = `${fastify.serverConfig.userPluginsDir}/${userId}/${name}`;
        let dirExists = false;
        try {
          await fsp.access(pluginDir);
          dirExists = true;
        } catch {
          // does not exist
        }

        if (dirExists) {
          await fastify.githubRepo.pull(pluginDir);
        } else {
          await fastify.githubRepo.cloneRepo(repo.cloneUrl, pluginDir);
          await fastify.githubRepo.configureGitUser(
            pluginDir,
            user!.githubLogin,
          );
          await fastify.userPluginStore.copyTemplateInto(userId, name);
          await fastify.githubRepo.commitAndPush(
            pluginDir,
            ghToken,
            user!.githubLogin,
            repo.cloneUrl,
            "Initial plugin from tidal template",
          );
        }
      } else {
        // No GitHub token — create plugin files locally only
        await fastify.userPluginStore.createFromTemplate(userId, name);
      }

      const plugin = fastify.db.createUserPlugin({
        userId,
        name,
        displayName,
        githubRepoUrl,
      });

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

  // -------------------------------------------------------------------------
  // Git sync endpoints
  // -------------------------------------------------------------------------

  /** GET /api/user-plugins/:id/git/status — git status. */
  fastify.get<{ Params: { id: string } }>(
    "/api/user-plugins/:id/git/status",
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

      const pluginDir = path.join(
        fastify.serverConfig.userPluginsDir,
        userId,
        plugin.name,
      );
      const status = await fastify.githubRepo.getStatus(pluginDir);
      return status;
    },
  );

  /** POST /api/user-plugins/:id/git/pull — pull latest from origin. */
  fastify.post<{ Params: { id: string } }>(
    "/api/user-plugins/:id/git/pull",
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

      const pluginDir = path.join(
        fastify.serverConfig.userPluginsDir,
        userId,
        plugin.name,
      );
      await fastify.githubRepo.pull(pluginDir);
      return { ok: true };
    },
  );

  /** POST /api/user-plugins/:id/git/push — commit + push changes. */
  fastify.post<{ Params: { id: string }; Body: PluginGitPushRequest }>(
    "/api/user-plugins/:id/git/push",
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

      const user = fastify.db.getUserById(userId);
      if (!user?.githubAccessToken) {
        return reply
          .status(400)
          .send({ error: "No GitHub token — cannot push" });
      }

      if (!plugin.githubRepoUrl) {
        return reply
          .status(400)
          .send({ error: "Plugin has no GitHub repo" });
      }

      const pluginDir = path.join(
        fastify.serverConfig.userPluginsDir,
        userId,
        plugin.name,
      );
      const repoCloneUrl = plugin.githubRepoUrl.endsWith(".git")
        ? plugin.githubRepoUrl
        : `${plugin.githubRepoUrl}.git`;

      await fastify.githubRepo.commitAndPush(
        pluginDir,
        user.githubAccessToken,
        user.githubLogin,
        repoCloneUrl,
        request.body.message,
      );
      return { ok: true };
    },
  );

  // -------------------------------------------------------------------------
  // Manifest endpoint
  // -------------------------------------------------------------------------

  /** GET /api/user-plugins/:id/manifest — read and parse manifest.yaml. */
  fastify.get<{ Params: { id: string } }>(
    "/api/user-plugins/:id/manifest",
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

      try {
        const content = await fastify.userPluginStore.readFile(
          userId,
          plugin.name,
          "manifest.yaml",
        );
        const manifest = parseYaml(content);
        return { manifest };
      } catch {
        return reply
          .status(404)
          .send({ error: "manifest.yaml not found" });
      }
    },
  );
}
