import path from "node:path";
import type { FastifyInstance } from "fastify";
import { ModelSourceBrowser } from "../services/model-source-browser.js";
import type { PluginFileTreeResponse, PluginFileReadResponse } from "@tidal/shared";

export default async function modelSourceRoutes(fastify: FastifyInstance) {
  const sourceDir = path.join(
    fastify.serverConfig.projectRoot,
    "plugins",
    "tidal",
  );
  const browser = new ModelSourceBrowser(sourceDir);

  /** GET /api/model/files — file tree of tidal model source. */
  fastify.get<{ Reply: PluginFileTreeResponse }>(
    "/api/model/files",
    async () => {
      const files = await browser.getFileTree();
      return { files };
    },
  );

  /** GET /api/model/files/* — read a specific source file. */
  fastify.get<{ Params: { "*": string }; Reply: PluginFileReadResponse }>(
    "/api/model/files/*",
    async (request, reply) => {
      const filePath = request.params["*"];

      if (!filePath) {
        return reply.status(400).send({ error: "File path is required" } as never);
      }

      try {
        const content = await browser.readFile(filePath);
        return { path: filePath, content };
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        if (message.includes("traversal") || message.includes("outside") || message.includes("absolute")) {
          return reply.status(400).send({ error: "Invalid file path" } as never);
        }
        return reply.status(404).send({ error: message } as never);
      }
    },
  );
}
