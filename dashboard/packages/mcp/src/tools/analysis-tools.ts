import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { TidalApiClient } from "../http-client.js";
import { jsonResult, errorResult, type CallToolResult } from "../tool-result.js";
import { CURATED_PROMPTS, type AnalyzeResponse } from "@tidal/shared";

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

export async function handleAnalyzeTrajectories(
  client: TidalApiClient,
  params: {
    checkpoint: string;
    prompts?: string[];
    maxTokens?: number;
    samplesPerPrompt?: number;
    gatingMode?: string;
    rlCheckpoint?: string;
    includeExtremeValues?: boolean;
  },
): Promise<CallToolResult> {
  const body = {
    checkpoint: params.checkpoint,
    prompts: params.prompts ?? CURATED_PROMPTS,
    maxTokens: params.maxTokens ?? 50,
    samplesPerPrompt: params.samplesPerPrompt ?? 1,
    gatingMode: params.gatingMode ?? "fixed",
    rlCheckpoint: params.rlCheckpoint,
    includeExtremeValues: params.includeExtremeValues ?? false,
  };

  const res = await client.post<AnalyzeResponse>("/api/analyze-trajectories", body);
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

export function registerAnalysisTools(
  server: McpServer,
  client: TidalApiClient,
): void {
  server.registerTool("analyze_trajectories", {
    description:
      "Analyze gate signal trajectories across multiple prompts. " +
      "Returns per-signal statistics, phase detection, cross-prompt variance, " +
      "and optionally an extreme-value sweep across 15 gate configurations.",
    inputSchema: {
      checkpoint: z.string().describe("Path to model checkpoint file"),
      prompts: z
        .array(z.string())
        .optional()
        .describe(
          "Prompts to generate from (default: 20 curated TinyStories prompts)",
        ),
      maxTokens: z
        .number()
        .optional()
        .describe("Maximum tokens to generate per prompt (default: 50)"),
      samplesPerPrompt: z
        .number()
        .optional()
        .describe("Number of generation samples per prompt (default: 1)"),
      gatingMode: z
        .enum(["random", "fixed", "learned"])
        .optional()
        .describe("Gating mode for generation (default: 'fixed')"),
      rlCheckpoint: z
        .string()
        .optional()
        .describe("Path to RL agent checkpoint (required when gatingMode is 'learned')"),
      includeExtremeValues: z
        .boolean()
        .optional()
        .describe(
          "Run a 15-config extreme-value sweep to map gate signal effects (default: false)",
        ),
    },
  }, async (params) => handleAnalyzeTrajectories(client, params));
}
