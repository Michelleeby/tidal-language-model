#!/usr/bin/env node

// ---------------------------------------------------------------------------
// Plan Review MCP Server — stdio transport for Claude Code integration
// Sends plans to GPT-4o, Gemini Flash, and Claude Sonnet for structured
// review across 5 dimensions, then deduplicates and ranks feedback.
// All logging goes to stderr (stdout is reserved for MCP JSON-RPC).
// ---------------------------------------------------------------------------

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { ProviderRegistry } from "./providers/provider.js";
import { OpenAIProvider } from "./providers/openai-provider.js";
import { GoogleProvider } from "./providers/google-provider.js";
import { AnthropicProvider } from "./providers/anthropic-provider.js";
import { registerReviewTools } from "./tools/review-tools.js";

const server = new McpServer({
  name: "plan-review",
  version: "0.1.0",
});

// Build provider registry — each checks its own env var
const registry = new ProviderRegistry();
registry.register(new OpenAIProvider());
registry.register(new GoogleProvider());
registry.register(new AnthropicProvider());

const available = registry.available();
if (available.length === 0) {
  console.error(
    "Warning: No API keys configured. Set OPENAI_API_KEY, GOOGLE_AI_API_KEY, or ANTHROPIC_API_KEY.",
  );
} else {
  console.error(
    `Plan Review providers: ${available.map((p) => p.name).join(", ")}`,
  );
}

// ADR directory from env or default
const adrDir = process.env.TIDAL_ADR_DIR ?? null;

// Register all 6 tools
registerReviewTools(server, registry, adrDir);

// Connect via stdio
const transport = new StdioServerTransport();
await server.connect(transport);

console.error("Plan Review MCP server running");
