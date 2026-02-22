// ---------------------------------------------------------------------------
// Tool handlers + MCP registration for code review
// ---------------------------------------------------------------------------

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { jsonResult, errorResult, type CallToolResult } from "../tool-result.js";
import type {
  FeedbackItem,
  CodeReviewDimension,
  ProviderName,
  Budget,
} from "../types.js";
import { ProviderRegistry } from "../providers/provider.js";
import {
  getCodeReviewDimensionPrompt,
  getAllCodeReviewDimensions,
} from "../prompts/code-review-prompts.js";
import { compactContext } from "../context/context-compactor.js";
import { aggregateFeedback } from "../feedback/feedback-aggregator.js";
import { summarizeADRs } from "../adr/adr-summarizer.js";
import { selectRelevantADRs } from "../adr/adr-relevance.js";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

// ---------------------------------------------------------------------------
// Model-to-dimension mapping per budget tier (code review specific)
// ---------------------------------------------------------------------------

type CodeReviewDimensionAssignment = {
  dimension: CodeReviewDimension;
  providers: ProviderName[];
};

function assignCodeReviewProviders(
  dimensions: CodeReviewDimension[],
  available: ProviderName[],
  budget: Budget,
): CodeReviewDimensionAssignment[] {
  // bugs: GPT-4o is strong at defect detection; adr_compliance: Gemini Flash cheapest for structured comparison
  const STANDARD_MAP: Record<CodeReviewDimension, [ProviderName, ProviderName]> = {
    bugs: ["openai", "google"],
    hypothesis_alignment: ["openai", "anthropic"],
    adr_compliance: ["google", "openai"],
  };

  return dimensions.map((dim) => {
    let providers: ProviderName[];
    if (budget === "thorough") {
      providers = available;
    } else if (budget === "minimal") {
      const preferred = STANDARD_MAP[dim]?.[0] ?? available[0];
      providers = available.includes(preferred) ? [preferred] : [available[0]];
    } else {
      // standard
      const mapped = STANDARD_MAP[dim] ?? [available[0]];
      providers = mapped.filter((p) => available.includes(p));
      if (providers.length === 0) providers = [available[0]];
    }
    return { dimension: dim, providers };
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function loadADRSummariesForReview(
  adrDir: string | null,
  diffText: string,
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
    const relevant = selectRelevantADRs(summaries, diffText, 1500);
    return relevant.map((s) => s.raw).join("\n\n");
  } catch {
    return "";
  }
}

// ---------------------------------------------------------------------------
// Handlers — pure functions for testing
// ---------------------------------------------------------------------------

export async function handleReviewCode(
  registry: ProviderRegistry,
  params: {
    diff: string;
    context?: string;
    dimensions?: CodeReviewDimension[];
    budget?: Budget;
    include_adrs?: boolean;
    adrDir: string | null;
  },
): Promise<CallToolResult> {
  const available = registry.available();
  if (available.length === 0) {
    return errorResult(
      "No providers available. Set at least one API key (OPENAI_API_KEY, GOOGLE_AI_API_KEY, or ANTHROPIC_API_KEY).",
    );
  }

  const dimensions = params.dimensions ?? getAllCodeReviewDimensions();
  const budget = params.budget ?? "standard";
  const includeAdrs = params.include_adrs !== false;

  // Load ADR summaries if requested
  const adrSummaries = includeAdrs
    ? await loadADRSummariesForReview(params.adrDir, params.diff)
    : "";

  // Compact context — diff is passed as "plan" since compactContext just handles text
  const compacted = compactContext({
    plan: params.diff,
    adrSummaries,
    codeContext: params.context ?? "",
    tokenBudget: 6000,
  });

  const providerNames = available.map((p) => p.name);
  const assignments = assignCodeReviewProviders(dimensions, providerNames, budget);

  // Fire all review calls in parallel
  const allPromises: Array<
    Promise<{
      dimension: CodeReviewDimension;
      feedback: FeedbackItem[];
      provider: ProviderName;
    }>
  > = [];

  for (const assignment of assignments) {
    for (const provName of assignment.providers) {
      const provider = registry.get(provName);
      if (!provider?.available) continue;

      const systemPrompt = getCodeReviewDimensionPrompt(assignment.dimension);
      const userPrompt = compacted.toString();

      allPromises.push(
        provider
          .review({ systemPrompt, userPrompt, dimension: assignment.dimension })
          .then((response) => ({
            dimension: assignment.dimension,
            feedback: response.feedback,
            provider: provName,
          }))
          .catch(() => ({
            dimension: assignment.dimension,
            feedback: [],
            provider: provName,
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

// ---------------------------------------------------------------------------
// Registration — wires handler into McpServer
// ---------------------------------------------------------------------------

export function registerCodeReviewTools(
  server: McpServer,
  registry: ProviderRegistry,
  adrDir: string | null,
): void {
  server.registerTool(
    "review_code",
    {
      description:
        "Review a code diff by sending it to multiple AI models across 3 dimensions (bugs, hypothesis_alignment, adr_compliance). Returns deduplicated, ranked feedback. Use after implementing a plan to catch bugs, verify hypothesis alignment, and confirm ADR compliance.",
      inputSchema: {
        diff: z.string().describe("Git diff or code changes to review"),
        dimensions: z
          .array(z.enum(["bugs", "hypothesis_alignment", "adr_compliance"]))
          .optional()
          .describe("Dimensions to review (default: all 3)"),
        context: z
          .string()
          .optional()
          .describe("Additional codebase context (e.g., related file contents)"),
        include_adrs: z
          .boolean()
          .optional()
          .describe("Inject relevant ADR summaries as context (default: true)"),
        budget: z
          .enum(["minimal", "standard", "thorough"])
          .optional()
          .describe(
            "Budget tier: minimal (1 model/dim), standard (2 models/dim), thorough (all 3)",
          ),
      },
    },
    async (params) => handleReviewCode(registry, { ...params, adrDir }),
  );
}
