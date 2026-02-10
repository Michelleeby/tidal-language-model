import type { FastifyInstance } from "fastify";
import { GenerationBridge } from "../services/generation-bridge.js";
import type { GenerateRequest, GenerateResponse } from "@tidal/shared";

export default async function generateRoutes(fastify: FastifyInstance) {
  const bridge = new GenerationBridge(fastify.serverConfig);

  fastify.post<{
    Body: GenerateRequest;
    Reply: GenerateResponse;
  }>("/api/generate", async (request) => {
    return bridge.generate(request.body);
  });
}
