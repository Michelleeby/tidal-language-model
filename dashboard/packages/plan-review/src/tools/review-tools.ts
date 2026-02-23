// ---------------------------------------------------------------------------
// Tool handlers + MCP registration for plan review
// ---------------------------------------------------------------------------

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { jsonResult, errorResult, type CallToolResult } from "../tool-result.js";
import type {
  FeedbackItem,
  ReviewDimension,
  ProviderName,
  Budget,
  CostEstimate,
  ModelAssignment,
} from "../types.js";
import { ProviderRegistry } from "../providers/provider.js";
import type { LLMProvider } from "../providers/provider.js";
import { getDimensionPrompt, getAllDimensions } from "../prompts/dimension-prompts.js";
import { compactContext } from "../context/context-compactor.js";
import { aggregateFeedback } from "../feedback/feedback-aggregator.js";
import { summarizeADRs } from "../adr/adr-summarizer.js";
import { selectRelevantADRs } from "../adr/adr-relevance.js";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

// ---------------------------------------------------------------------------
// Model-to-dimension mapping per budget tier
// ---------------------------------------------------------------------------

type DimensionAssignment = { dimension: ReviewDimension; assignments: ModelAssignment[] };

const THOROUGH_ASSIGNMENTS: ModelAssignment[] = [
  { provider: "openai", model: "gpt-4o" },
  { provider: "openai", model: "o3-mini" },
  { provider: "google" },
];

export function assignProviders(
  dimensions: ReviewDimension[],
  available: ProviderName[],
  budget: Budget,
): DimensionAssignment[] {
  const STANDARD_MAP: Record<ReviewDimension, [ProviderName, ProviderName]> = {
    completeness: ["openai", "google"],
    blind_spots: ["openai", "google"],
    regression_risk: ["google", "openai"],
    test_coverage: ["google", "openai"],
    hypothesis_scope: ["google", "openai"],
  };

  return dimensions.map((dim) => {
    let assignments: ModelAssignment[];
    if (budget === "thorough") {
      assignments = THOROUGH_ASSIGNMENTS.filter((ma) => available.includes(ma.provider));
    } else if (budget === "minimal") {
      const preferred = STANDARD_MAP[dim]?.[0] ?? available[0];
      const provider = available.includes(preferred) ? preferred : available[0];
      assignments = [{ provider }];
    } else {
      // standard
      const mapped = STANDARD_MAP[dim] ?? [available[0]];
      const filtered = mapped.filter((p) => available.includes(p));
      const providers = filtered.length > 0 ? filtered : [available[0]];
      assignments = providers.map((provider) => ({ provider }));
    }
    return { dimension: dim, assignments };
  });
}

// ---------------------------------------------------------------------------
// Handlers — pure functions for testing
// ---------------------------------------------------------------------------

export async function handleListProviders(
  registry: ProviderRegistry,
): Promise<CallToolResult> {
  const all = registry.all();
  const available = all.filter((p) => p.available).map((p) => p.name);
  const unavailable = all.filter((p) => !p.available).map((p) => p.name);
  return jsonResult({ available, unavailable });
}

export async function handleGetReviewCosts(
  registry: ProviderRegistry,
  params: { plan: string; budget?: Budget },
): Promise<CallToolResult> {
  const budget = params.budget ?? "standard";
  const available = registry.available();
  if (available.length === 0) {
    return errorResult("No providers available. Set at least one API key.");
  }

  const planTokens = Math.ceil(params.plan.length / 4);
  const systemTokens = 500;
  const adrTokens = 1500;
  const responseTokens = 2000;
  const inputPerCall = systemTokens + adrTokens + planTokens;

  const dimensions = getAllDimensions();
  const assignments = assignProviders(
    dimensions,
    available.map((p) => p.name),
    budget,
  );

  const estimates: CostEstimate[] = [];
  const COST_PER_1K: Record<string, { input: number; output: number }> = {
    "gpt-4o": { input: 0.0025, output: 0.01 },
    "o3-mini": { input: 0.0011, output: 0.0044 },
    "gemini-2.0-flash": { input: 0.0001, output: 0.0004 },
    "mock-model": { input: 0.001, output: 0.004 },
  };

  for (const assignment of assignments) {
    for (const ma of assignment.assignments) {
      const provider = registry.get(ma.provider);
      if (!provider) continue;
      const model = ma.model ?? provider.models[0];
      const costs = COST_PER_1K[model] ?? { input: 0.003, output: 0.015 };
      estimates.push({
        provider: ma.provider,
        model,
        estimatedInputTokens: inputPerCall,
        estimatedOutputTokens: responseTokens,
        estimatedCostUsd:
          (inputPerCall / 1000) * costs.input +
          (responseTokens / 1000) * costs.output,
      });
    }
  }

  const totalEstimatedCostUsd = estimates.reduce(
    (sum, e) => sum + e.estimatedCostUsd,
    0,
  );

  return jsonResult({
    budget,
    dimensions: dimensions.length,
    totalCalls: estimates.length,
    estimates,
    totalEstimatedCostUsd: Math.round(totalEstimatedCostUsd * 10000) / 10000,
  });
}

export async function handleReviewDimension(
  registry: ProviderRegistry,
  params: {
    plan: string;
    dimension: ReviewDimension;
    provider: ProviderName;
    context?: string;
    model?: string;
  },
): Promise<CallToolResult> {
  const provider = registry.get(params.provider);
  if (!provider || !provider.available) {
    return errorResult(`Provider "${params.provider}" is not available.`);
  }

  const systemPrompt = getDimensionPrompt(params.dimension);
  const userPrompt = params.context
    ? `## Plan\n\n${params.plan}\n\n## Additional Context\n\n${params.context}`
    : `## Plan\n\n${params.plan}`;

  const response = await provider.review(
    { systemPrompt, userPrompt, dimension: params.dimension },
    params.model,
  );

  return jsonResult({
    feedback: response.feedback,
    provider: response.provider,
    model: response.model,
    dimension: response.dimension,
  });
}

async function loadADRSummaries(
  adrDir: string | null,
  planText: string,
): Promise<string> {
  if (!adrDir) return "";

  try {
    const files = await readdir(adrDir);
    const adrFiles = files.filter((f) => f.endsWith(".md")).sort();

    const entries = await Promise.all(
      adrFiles.map(async (filename) => ({
        filename,
        content: await readFile(join(adrDir, filename), "utf-8"),
      })),
    );

    const summaries = summarizeADRs(entries);
    const relevant = selectRelevantADRs(summaries, planText, 1500);
    return relevant.map((s) => s.raw).join("\n\n");
  } catch {
    return "";
  }
}

export async function handleReviewPlan(
  registry: ProviderRegistry,
  params: {
    plan: string;
    dimensions?: ReviewDimension[];
    context?: string;
    include_adrs?: boolean;
    providers?: ProviderName[];
    budget?: Budget;
    adrDir: string | null;
  },
): Promise<CallToolResult> {
  const available = registry.available();
  const requestedProviders = params.providers
    ? available.filter((p) => params.providers!.includes(p.name))
    : available;

  if (requestedProviders.length === 0) {
    return errorResult(
      "No providers available. Set at least one API key (OPENAI_API_KEY or GOOGLE_AI_API_KEY).",
    );
  }

  const dimensions = params.dimensions ?? getAllDimensions();
  const budget = params.budget ?? "standard";
  const includeAdrs = params.include_adrs !== false;

  // Load ADR summaries if requested
  const adrSummaries = includeAdrs
    ? await loadADRSummaries(params.adrDir, params.plan)
    : "";

  // Compact context for each prompt
  const compacted = compactContext({
    plan: params.plan,
    adrSummaries,
    codeContext: params.context ?? "",
    tokenBudget: 6000,
  });

  const providerNames = requestedProviders.map((p) => p.name);
  const assignments = assignProviders(dimensions, providerNames, budget);

  // Fire all review calls in parallel
  const allPromises: Array<
    Promise<{ dimension: ReviewDimension; feedback: FeedbackItem[]; provider: ProviderName }>
  > = [];

  for (const assignment of assignments) {
    for (const ma of assignment.assignments) {
      const provider = registry.get(ma.provider);
      if (!provider?.available) continue;

      const systemPrompt = getDimensionPrompt(assignment.dimension);
      const userPrompt = compacted.toString();

      allPromises.push(
        provider
          .review(
            { systemPrompt, userPrompt, dimension: assignment.dimension },
            ma.model,
          )
          .then((response) => ({
            dimension: assignment.dimension,
            feedback: response.feedback,
            provider: ma.provider,
          }))
          .catch(() => ({
            dimension: assignment.dimension,
            feedback: [],
            provider: ma.provider,
          })),
      );
    }
  }

  const results = await Promise.allSettled(allPromises);
  const allFeedback: FeedbackItem[] = [];
  const providersUsed = new Set<ProviderName>();

  for (const result of results) {
    if (result.status === "fulfilled") {
      allFeedback.push(...result.value.feedback);
      providersUsed.add(result.value.provider);
    }
  }

  // Aggregate: dedup + rank
  const aggregated = aggregateFeedback(allFeedback);

  return jsonResult({
    feedback: aggregated,
    dimensions_reviewed: dimensions,
    providers_used: [...providersUsed],
    budget,
    total_raw_items: allFeedback.length,
    total_aggregated_items: aggregated.length,
  });
}

export async function handleSummarizeADRs(params: {
  adrDir: string | null;
  keywords?: string[];
}): Promise<CallToolResult> {
  if (!params.adrDir) {
    return errorResult(
      "ADR directory not configured. Set TIDAL_ADR_DIR environment variable.",
    );
  }

  try {
    const files = await readdir(params.adrDir);
    const adrFiles = files.filter((f) => f.endsWith(".md")).sort();

    const entries = await Promise.all(
      adrFiles.map(async (filename) => ({
        filename,
        content: await readFile(join(params.adrDir!, filename), "utf-8"),
      })),
    );

    const summaries = summarizeADRs(entries);

    if (params.keywords?.length) {
      const filterText = params.keywords.join(" ");
      const relevant = selectRelevantADRs(summaries, filterText, 5000);
      return jsonResult({ summaries: relevant });
    }

    return jsonResult({ summaries });
  } catch (err) {
    return errorResult(`Failed to read ADR directory: ${err}`);
  }
}

export async function handleAggregateFeedback(params: {
  feedback: FeedbackItem[];
}): Promise<CallToolResult> {
  const aggregated = aggregateFeedback(params.feedback);
  return jsonResult({ feedback: aggregated });
}

// ---------------------------------------------------------------------------
// Registration — wires handlers into McpServer
// ---------------------------------------------------------------------------

export function registerReviewTools(
  server: McpServer,
  registry: ProviderRegistry,
  adrDir: string | null,
): void {
  server.registerTool("review_plan", {
    description:
      "Review an implementation plan by sending it to multiple AI models across 5 dimensions (completeness, blind_spots, regression_risk, test_coverage, hypothesis_scope). Returns deduplicated, ranked feedback.",
    inputSchema: {
      plan: z.string().describe("Full plan text to review"),
      dimensions: z
        .array(
          z.enum([
            "completeness",
            "blind_spots",
            "regression_risk",
            "test_coverage",
            "hypothesis_scope",
          ]),
        )
        .optional()
        .describe("Dimensions to review (default: all 5)"),
      context: z
        .string()
        .optional()
        .describe("Additional codebase context"),
      include_adrs: z
        .boolean()
        .optional()
        .describe("Inject relevant ADR summaries as context (default: true)"),
      providers: z
        .array(z.enum(["openai", "google"]))
        .optional()
        .describe("Providers to use (default: all available)"),
      budget: z
        .enum(["minimal", "standard", "thorough"])
        .optional()
        .describe(
          "Budget tier: minimal (1 model/dim), standard (2 models/dim), thorough (all 3)",
        ),
    },
  }, async (params) =>
    handleReviewPlan(registry, { ...params, adrDir }),
  );

  server.registerTool("review_dimension", {
    description:
      "Review a plan on a single dimension with a single provider. For targeted follow-up on specific concerns.",
    inputSchema: {
      plan: z.string().describe("Full plan text to review"),
      dimension: z
        .enum([
          "completeness",
          "blind_spots",
          "regression_risk",
          "test_coverage",
          "hypothesis_scope",
        ])
        .describe("The review dimension"),
      provider: z
        .enum(["openai", "google"])
        .describe("The provider to use"),
      context: z.string().optional().describe("Additional context"),
      model: z.string().optional().describe("Specific model to use"),
    },
  }, async (params) => handleReviewDimension(registry, params));

  server.registerTool("summarize_adrs", {
    description:
      "Return compact ADR summaries from the project's architecture decision records. Optionally filter by domain keywords.",
    inputSchema: {
      keywords: z
        .array(z.string())
        .optional()
        .describe("Domain keywords to filter relevant ADRs"),
    },
  }, async (params) => handleSummarizeADRs({ ...params, adrDir }));

  server.registerTool("aggregate_feedback", {
    description:
      "Standalone dedup and ranking for raw feedback arrays. Uses Jaccard similarity for dedup and severity-based scoring for ranking.",
    inputSchema: {
      feedback: z
        .array(
          z.object({
            severity: z.enum(["critical", "warning", "suggestion"]),
            description: z.string(),
            affected_files: z.array(z.string()),
            reasoning: z.string(),
            dimension: z.enum([
              "completeness",
              "blind_spots",
              "regression_risk",
              "test_coverage",
              "hypothesis_scope",
            ]),
            source: z.enum(["openai", "google"]),
          }),
        )
        .describe("Raw feedback items to aggregate"),
    },
  }, async (params) => handleAggregateFeedback(params));

  server.registerTool("list_review_providers", {
    description:
      "Show which AI review providers are configured and available (have valid API keys).",
    inputSchema: {},
  }, async () => handleListProviders(registry));

  server.registerTool("get_review_costs", {
    description:
      "Estimate token usage and cost before executing a review. Helps choose a budget tier.",
    inputSchema: {
      plan: z.string().describe("Plan text to estimate costs for"),
      budget: z
        .enum(["minimal", "standard", "thorough"])
        .optional()
        .describe("Budget tier to estimate (default: standard)"),
    },
  }, async (params) => handleGetReviewCosts(registry, params));
}
