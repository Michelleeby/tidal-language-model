import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { TidalApiClient } from "../http-client.js";
import { jsonResult, errorResult, type CallToolResult } from "../tool-result.js";
import type { JobsListResponse, JobLogsResponse } from "@tidal/shared";

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
}
