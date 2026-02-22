import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { TidalApiClient } from "../http-client.js";
import { jsonResult, errorResult, type CallToolResult } from "../tool-result.js";
import type {
  JobsListResponse,
  JobLogsResponse,
  CreateJobRequest,
  CreateJobResponse,
} from "@tidal/shared";

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

export async function handleGetJobs(
  client: TidalApiClient,
): Promise<CallToolResult> {
  const res = await client.get<JobsListResponse>("/api/jobs");
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

export async function handleGetJobLogs(
  client: TidalApiClient,
  params: { jobId: string; offset?: number; limit?: number },
): Promise<CallToolResult> {
  const res = await client.get<JobLogsResponse>(
    `/api/jobs/${params.jobId}/logs`,
    { offset: params.offset, limit: params.limit },
  );
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

export async function handleCreateJob(
  client: TidalApiClient,
  params: {
    type: string;
    configPath: string;
    overlayConfigPath?: string;
    provider?: "local" | "vastai";
  },
): Promise<CallToolResult> {
  const body: CreateJobRequest = {
    type: params.type,
    plugin: "tidal",
    configPath: params.configPath,
    overlayConfigPath: params.overlayConfigPath,
    provider: params.provider,
  };
  const res = await client.post<CreateJobResponse>("/api/jobs", body);
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

export function registerJobTools(
  server: McpServer,
  client: TidalApiClient,
): void {
  server.registerTool("get_jobs", {
    description: "List all training jobs with status, provider, and config",
    inputSchema: {},
  }, async () => handleGetJobs(client));

  server.registerTool("get_job_logs", {
    description: "Get stdout/stderr logs for a training job",
    inputSchema: {
      jobId: z.string().describe("Job ID"),
      offset: z.number().optional().describe("Line offset to start from"),
      limit: z.number().optional().describe("Maximum number of log lines to return"),
    },
  }, async (params) => handleGetJobLogs(client, params));

  server.registerTool("create_job", {
    description: "Create and launch a new training job on local or Vast.ai GPU",
    inputSchema: {
      type: z.string().describe("Job type (e.g. 'lm-training', 'lm-experiment')"),
      configPath: z.string().describe("Config file path relative to plugin directory"),
      overlayConfigPath: z.string().optional().describe("Overlay config path merged on top of configPath"),
      provider: z.enum(["local", "vastai"]).optional().describe("Compute provider (default: local)"),
    },
  }, async (params) => handleCreateJob(client, params));
}
