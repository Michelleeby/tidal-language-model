// ---------------------------------------------------------------------------
// Shared CallToolResult helpers — compatible with MCP SDK's index signature
// ---------------------------------------------------------------------------

export interface CallToolResult {
  [key: string]: unknown;
  content: Array<{ type: "text"; text: string }>;
  isError?: boolean;
}

export function jsonResult(data: unknown): CallToolResult {
  return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
}

export function errorResult(error: string): CallToolResult {
  return { content: [{ type: "text", text: `Error: ${error}` }], isError: true };
}
