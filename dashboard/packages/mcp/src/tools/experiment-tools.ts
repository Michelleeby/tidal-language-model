import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { TidalApiClient } from "../http-client.js";
import { jsonResult, errorResult, type CallToolResult } from "../tool-result.js";
import type {
  ExperimentsResponse,
  MetricsResponse,
  RLMetricsResponse,
  StatusResponse,
  CheckpointsResponse,
  EvaluationResponse,
  AblationResponse,
} from "@tidal/shared";

// ---------------------------------------------------------------------------
// Handlers — pure functions for direct unit testing
// ---------------------------------------------------------------------------

export async function handleListExperiments(
  client: TidalApiClient,
): Promise<CallToolResult> {
  const res = await client.get<ExperimentsResponse>("/api/experiments");
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

export async function handleGetMetrics(
  client: TidalApiClient,
  params: { expId: string; mode?: string; window?: number; maxPoints?: number },
): Promise<CallToolResult> {
  const res = await client.get<MetricsResponse>(
    `/api/experiments/${params.expId}/metrics`,
    { mode: params.mode, window: params.window, maxPoints: params.maxPoints },
  );
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

export async function handleGetRlMetrics(
  client: TidalApiClient,
  params: { expId: string },
): Promise<CallToolResult> {
  const res = await client.get<RLMetricsResponse>(
    `/api/experiments/${params.expId}/rl-metrics`,
  );
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

export async function handleGetStatus(
  client: TidalApiClient,
  params: { expId: string },
): Promise<CallToolResult> {
  const res = await client.get<StatusResponse>(
    `/api/experiments/${params.expId}/status`,
  );
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

export async function handleGetCheckpoints(
  client: TidalApiClient,
  params: { expId: string },
): Promise<CallToolResult> {
  const res = await client.get<CheckpointsResponse>(
    `/api/experiments/${params.expId}/checkpoints`,
  );
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

export async function handleGetEvaluation(
  client: TidalApiClient,
  params: { expId: string },
): Promise<CallToolResult> {
  const res = await client.get<EvaluationResponse>(
    `/api/experiments/${params.expId}/evaluation`,
  );
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

export async function handleGetAblation(
  client: TidalApiClient,
  params: { expId: string },
): Promise<CallToolResult> {
  const res = await client.get<AblationResponse>(
    `/api/experiments/${params.expId}/ablation`,
  );
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

// ---------------------------------------------------------------------------
// Registration — wires handlers into McpServer
// ---------------------------------------------------------------------------

export function registerExperimentTools(
  server: McpServer,
  client: TidalApiClient,
): void {
  server.registerTool("list_experiments", {
    description:
      "List all Tidal training experiments with IDs, status, and data availability flags",
    inputSchema: {},
  }, async () => handleListExperiments(client));

  server.registerTool("get_metrics", {
    description:
      "Get LM training metrics (loss, learning rate) for an experiment",
    inputSchema: {
      expId: z.string().describe("Experiment ID"),
      mode: z
        .enum(["recent", "historical"])
        .optional()
        .describe("Retrieval mode: 'recent' for latest window, 'historical' for all"),
      window: z.number().optional().describe("Number of recent points to return"),
      maxPoints: z
        .number()
        .optional()
        .describe("Maximum points (downsamples if exceeded)"),
    },
  }, async (params) => handleGetMetrics(client, params));

  server.registerTool("get_rl_metrics", {
    description:
      "Get RL gating controller metrics (policy loss, gate signals, rewards) for an experiment",
    inputSchema: {
      expId: z.string().describe("Experiment ID"),
    },
  }, async (params) => handleGetRlMetrics(client, params));

  server.registerTool("get_status", {
    description: "Get training status and progress for an experiment",
    inputSchema: {
      expId: z.string().describe("Experiment ID"),
    },
  }, async (params) => handleGetStatus(client, params));

  server.registerTool("get_checkpoints", {
    description:
      "List available checkpoints with file sizes for an experiment",
    inputSchema: {
      expId: z.string().describe("Experiment ID"),
    },
  }, async (params) => handleGetCheckpoints(client, params));

  server.registerTool("get_evaluation", {
    description:
      "Get evaluation results (perplexity + generated samples) for an experiment",
    inputSchema: {
      expId: z.string().describe("Experiment ID"),
    },
  }, async (params) => handleGetEvaluation(client, params));

  server.registerTool("get_ablation", {
    description:
      "Get ablation study comparisons for an experiment",
    inputSchema: {
      expId: z.string().describe("Experiment ID"),
    },
  }, async (params) => handleGetAblation(client, params));
}
