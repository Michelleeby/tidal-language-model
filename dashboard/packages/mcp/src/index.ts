#!/usr/bin/env node

// ---------------------------------------------------------------------------
// Tidal MCP Server — stdio transport for Claude Code integration
// All logging goes to stderr (stdout is reserved for MCP JSON-RPC).
// ---------------------------------------------------------------------------

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { FetchTidalApiClient } from "./http-client.js";
import { registerExperimentTools } from "./tools/experiment-tools.js";
import { registerPluginTools } from "./tools/plugin-tools.js";
import { registerJobTools } from "./tools/job-tools.js";
import { registerGenerationTools } from "./tools/generation-tools.js";
import { registerAnalysisTools } from "./tools/analysis-tools.js";
import { registerReportTools } from "./tools/report-tools.js";

const BASE_URL = process.env.TIDAL_API_URL ?? "http://localhost:4400";
const TOKEN = process.env.TIDAL_API_TOKEN;

const server = new McpServer({
  name: "tidal-dashboard",
  version: "0.1.0",
});

const client = new FetchTidalApiClient(BASE_URL, TOKEN);

// Register all 15 tools
registerExperimentTools(server, client);
registerPluginTools(server, client);
registerJobTools(server, client);
registerGenerationTools(server, client);
registerAnalysisTools(server, client);
registerReportTools(server, client);

// Connect via stdio
const transport = new StdioServerTransport();
await server.connect(transport);

console.error(`Tidal MCP server running (API: ${BASE_URL})`);
