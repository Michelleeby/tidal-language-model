import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { TidalApiClient } from "../http-client.js";
import { jsonResult, errorResult, type CallToolResult } from "../tool-result.js";
import { listPatterns } from "@tidal/shared";
import type { GenerateReportResponse, ReportResponse } from "@tidal/shared";

// ---------------------------------------------------------------------------
// Handlers — pure functions for direct unit testing
// ---------------------------------------------------------------------------

export async function handleListPatterns(): Promise<CallToolResult> {
  return jsonResult({ patterns: listPatterns() });
}

export async function handleGenerateReport(
  client: TidalApiClient,
  params: {
    pattern: string;
    experimentId: string;
    title?: string;
    githubLogin?: string;
  },
): Promise<CallToolResult> {
  const res = await client.post<GenerateReportResponse>(
    "/api/reports/generate",
    {
      pattern: params.pattern,
      experimentId: params.experimentId,
      ...(params.title !== undefined && { title: params.title }),
      ...(params.githubLogin !== undefined && { githubLogin: params.githubLogin }),
    },
  );
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

export async function handleUpdateReport(
  client: TidalApiClient,
  params: {
    reportId: string;
    title?: string;
    blocks?: Record<string, unknown>[];
  },
): Promise<CallToolResult> {
  const res = await client.put<ReportResponse>(
    `/api/reports/${params.reportId}`,
    {
      ...(params.title !== undefined && { title: params.title }),
      ...(params.blocks !== undefined && { blocks: params.blocks }),
    },
  );
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

// ---------------------------------------------------------------------------
// Registration — wires handlers into McpServer
// ---------------------------------------------------------------------------

export function registerReportTools(
  server: McpServer,
  client: TidalApiClient,
): void {
  server.registerTool("list_patterns", {
    description:
      "List available block patterns for report generation. Returns pattern names and descriptions.",
    inputSchema: {},
  }, async () => handleListPatterns());

  server.registerTool("generate_report", {
    description:
      "Generate a pre-populated report from a block pattern and experiment ID. Creates a report with charts, tables, and headings ready to view in the dashboard.",
    inputSchema: {
      pattern: z
        .string()
        .describe("Pattern name: 'experiment-overview', 'rl-analysis', 'trajectory-report', or 'full-report'"),
      experimentId: z.string().describe("Experiment ID to populate the report with"),
      title: z.string().optional().describe("Custom report title (defaults to pattern + experiment ID)"),
      githubLogin: z
        .string()
        .optional()
        .describe("GitHub login to associate the report with a user"),
    },
  }, async (params) => handleGenerateReport(client, params));

  server.registerTool("update_report", {
    description:
      "Update an existing report's title and/or blocks. Use after generate_report to replace scaffolded blocks with enriched content including analysis paragraphs.",
    inputSchema: {
      reportId: z.string().describe("The report ID to update"),
      title: z.string().optional().describe("New report title"),
      blocks: z
        .array(z.record(z.unknown()))
        .optional()
        .describe("New BlockNote blocks array (replaces existing blocks)"),
    },
  }, async (params) => handleUpdateReport(client, params));
}
