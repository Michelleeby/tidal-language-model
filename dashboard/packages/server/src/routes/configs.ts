import type { FastifyInstance } from "fastify";
import fsp from "node:fs/promises";
import path from "node:path";
import type { ConfigFileResponse, ConfigListResponse } from "@tidal/shared";

const SAFE_FILENAME = /^[a-zA-Z0-9_-]+\.(yaml|yml)$/;

export default async function configsRoutes(fastify: FastifyInstance) {
  const pluginsDir = path.join(fastify.serverConfig.projectRoot, "plugins");

  /** GET /api/plugins/:name/configs — list YAML config files for a plugin. */
  fastify.get<{ Params: { name: string }; Reply: ConfigListResponse | { error: string } }>(
    "/api/plugins/:name/configs",
    async (request, reply) => {
      const { name } = request.params;
      const configDir = path.join(pluginsDir, name, "configs");

      try {
        const entries = await fsp.readdir(configDir);
        const yamlFiles = entries.filter((f) => /\.(yaml|yml)$/.test(f)).sort();
        return { plugin: name, files: yamlFiles };
      } catch {
        return reply.status(404).send({ error: "Plugin config directory not found" });
      }
    },
  );

  /** GET /api/plugins/:name/configs/:filename — read a single YAML config file. */
  fastify.get<{
    Params: { name: string; filename: string };
    Reply: ConfigFileResponse | { error: string };
  }>(
    "/api/plugins/:name/configs/:filename",
    async (request, reply) => {
      const { name, filename } = request.params;

      // Validate filename: alphanumeric + hyphens/underscores + .yaml/.yml only
      if (!SAFE_FILENAME.test(filename)) {
        return reply.status(400).send({ error: "Invalid filename" });
      }

      const configDir = path.join(pluginsDir, name, "configs");
      const filePath = path.join(configDir, filename);

      // Prevent path traversal: resolved path must be inside configDir
      const resolved = path.resolve(filePath);
      if (!resolved.startsWith(path.resolve(configDir) + path.sep)) {
        return reply.status(400).send({ error: "Invalid filename" });
      }

      try {
        const content = await fsp.readFile(resolved, "utf-8");
        return { filename, content };
      } catch {
        return reply.status(404).send({ error: "Config file not found" });
      }
    },
  );
}
