import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { TidalApiClient } from "../http-client.js";
import { jsonResult, errorResult, type CallToolResult } from "../tool-result.js";
import type { GenerateResponse } from "@tidal/shared";

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

export async function handleGenerateText(
  client: TidalApiClient,
  params: {
    checkpoint: string;
    prompt: string;
    maxTokens?: number;
    temperature?: number;
    topK?: number;
    gatingMode?: string;
    rlCheckpoint?: string;
    creativity?: number;
    focus?: number;
    stability?: number;
  },
): Promise<CallToolResult> {
  const res = await client.post<GenerateResponse>("/api/generate", params);
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

export function registerGenerationTools(
  server: McpServer,
  client: TidalApiClient,
): void {
  server.registerTool("generate_text", {
    description:
      "Generate text using a trained Tidal model checkpoint, optionally with RL-controlled gating",
    inputSchema: {
      checkpoint: z.string().describe("Path to model checkpoint file"),
      prompt: z.string().describe("Text prompt to continue from"),
      maxTokens: z.number().optional().describe("Maximum tokens to generate (default: 50)"),
      temperature: z.number().optional().describe("Sampling temperature (default: 0.8)"),
      topK: z.number().optional().describe("Top-k sampling parameter (default: 50)"),
      gatingMode: z
        .enum(["none", "random", "fixed", "learned"])
        .optional()
        .describe("Gating mode: 'none', 'random', 'fixed', or 'learned'"),
      rlCheckpoint: z
        .string()
        .optional()
        .describe("Path to RL agent checkpoint (required when gatingMode is 'learned')"),
      creativity: z
        .number()
        .optional()
        .describe("Creativity gate signal [0-1] for fixed gating mode (default: 0.5)"),
      focus: z
        .number()
        .optional()
        .describe("Focus gate signal [0-1] for fixed gating mode (default: 0.5)"),
      stability: z
        .number()
        .optional()
        .describe("Stability gate signal [0-1] for fixed gating mode (default: 0.5)"),
    },
  }, async (params) => handleGenerateText(client, params));
}
