import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { TidalApiClient } from "../http-client.js";
import { jsonResult, errorResult, type CallToolResult } from "../tool-result.js";
import type {
  PluginsListResponse,
  PluginResponse,
  ConfigFileResponse,
} from "@tidal/shared";

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

export async function handleListPlugins(
  client: TidalApiClient,
): Promise<CallToolResult> {
  const res = await client.get<PluginsListResponse>("/api/plugins");
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

export async function handleGetPluginManifest(
  client: TidalApiClient,
  params: { pluginName?: string },
): Promise<CallToolResult> {
  const name = params.pluginName ?? "tidal";
  const res = await client.get<PluginResponse>(`/api/plugins/${name}`);
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

export async function handleGetConfig(
  client: TidalApiClient,
  params: { pluginName?: string; filename: string },
): Promise<CallToolResult> {
  const name = params.pluginName ?? "tidal";
  const res = await client.get<ConfigFileResponse>(
    `/api/plugins/${name}/configs/${params.filename}`,
  );
  return res.ok ? jsonResult(res.data) : errorResult(res.error);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

export function registerPluginTools(
  server: McpServer,
  client: TidalApiClient,
): void {
  server.registerTool("list_plugins", {
    description: "List model plugins (currently only the tidal model)",
    inputSchema: {},
  }, async () => handleListPlugins(client));

  server.registerTool("get_plugin_manifest", {
    description: "Get the full manifest for a model plugin (training phases, checkpoint patterns, generation config, metrics config)",
    inputSchema: {
      pluginName: z.string().optional().describe("Plugin name (defaults to 'tidal')"),
    },
  }, async (params) => handleGetPluginManifest(client, params));

  server.registerTool("get_config", {
    description: "Read a YAML config file for a plugin (e.g. base_config.yaml, rl_config.yaml)",
    inputSchema: {
      pluginName: z.string().optional().describe("Plugin name (defaults to 'tidal')"),
      filename: z.string().describe("Config filename (e.g. 'base_config.yaml')"),
    },
  }, async (params) => handleGetConfig(client, params));
}
