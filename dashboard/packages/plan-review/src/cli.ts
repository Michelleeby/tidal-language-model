#!/usr/bin/env node

// ---------------------------------------------------------------------------
// CLI entry point — runs the review pipeline without an LLM orchestrator
//
// Usage:
//   node dist/cli.js review-pr <N> [--output path] [--post-comment] [--budget tier]
//   node dist/cli.js review-plan <file> [--output path] [--budget tier]
// ---------------------------------------------------------------------------

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { ProviderRegistry } from "./providers/provider.js";
import { OpenAIProvider } from "./providers/openai-provider.js";
import { GoogleProvider } from "./providers/google-provider.js";
import { handleReviewCode } from "./tools/code-review-tools.js";
import { handleReviewPlan } from "./tools/review-tools.js";
import { formatReviewReport } from "./report-formatter.js";
import type { ProviderName, Budget, AggregatedFeedbackItem } from "./types.js";

// ---------------------------------------------------------------------------
// Dependency injection interface for testability
// ---------------------------------------------------------------------------

export interface CLIDeps {
  execGH: (args: string[]) => string;
  readFileSync: (path: string) => string;
  writeFileSync: (path: string, content: string) => void;
  writtenFiles: Record<string, string>;
  registry: ProviderRegistry;
  adrDir: string | null;
}

interface ReviewOptions {
  output?: string;
  postComment?: boolean;
  budget?: Budget;
}

function createDefaultDeps(): CLIDeps {
  const registry = new ProviderRegistry();
  registry.register(new OpenAIProvider());
  registry.register(new GoogleProvider());

  const writtenFiles: Record<string, string> = {};
  return {
    execGH: (args: string[]) =>
      execFileSync("gh", args, { encoding: "utf-8", maxBuffer: 10 * 1024 * 1024 }),
    readFileSync: (path: string) => readFileSync(path, "utf-8"),
    writeFileSync: (path: string, content: string) => {
      writeFileSync(path, content, "utf-8");
      writtenFiles[path] = content;
    },
    writtenFiles,
    registry,
    adrDir: process.env.TIDAL_ADR_DIR ?? null,
  };
}

// ---------------------------------------------------------------------------
// Core review functions
// ---------------------------------------------------------------------------

export async function reviewPR(
  prNumber: string,
  options: ReviewOptions,
  deps: CLIDeps,
): Promise<string> {
  // Get diff
  let diff: string;
  try {
    diff = deps.execGH(["pr", "diff", prNumber]);
  } catch (err) {
    throw new Error(`Failed to get PR diff via gh: ${(err as Error).message}`);
  }

  // Get PR context
  let prContext = "";
  try {
    const prJson = deps.execGH(["pr", "view", prNumber, "--json", "title,body"]);
    const pr = JSON.parse(prJson);
    prContext = `PR #${prNumber}: ${pr.title}\n\n${pr.body ?? ""}`;
  } catch {
    // PR context is optional — continue without it
  }

  // Run review
  const budget = options.budget ?? "standard";
  const result = await handleReviewCode(deps.registry, {
    diff,
    context: prContext,
    budget,
    adrDir: deps.adrDir,
  });

  if (result.isError) {
    throw new Error(result.content[0].text as string);
  }

  const parsed = JSON.parse(result.content[0].text as string);
  const report = formatReviewReport(
    parsed.feedback as AggregatedFeedbackItem[],
    {
      dimensions: parsed.dimensions_reviewed,
      providers: parsed.providers_used as ProviderName[],
      budget,
    },
  );

  // Write output
  if (options.output) {
    deps.writeFileSync(options.output, report);
  }

  // Post comment
  if (options.postComment) {
    const tmpPath = options.output ?? `/tmp/review-${prNumber}.md`;
    if (!options.output) {
      deps.writeFileSync(tmpPath, report);
    }
    try {
      deps.execGH(["pr", "comment", prNumber, "--body-file", tmpPath]);
    } catch (err) {
      console.error(`Warning: Failed to post PR comment: ${(err as Error).message}`);
    }
  }

  return report;
}

export async function reviewPlan(
  planPath: string,
  options: ReviewOptions,
  deps: CLIDeps,
): Promise<string> {
  // Read plan file
  let planText: string;
  try {
    planText = deps.readFileSync(planPath);
  } catch (err) {
    throw new Error(`Failed to read plan file: ${(err as Error).message}`);
  }

  // Run review
  const budget = options.budget ?? "standard";
  const result = await handleReviewPlan(deps.registry, {
    plan: planText,
    budget,
    adrDir: deps.adrDir,
  });

  if (result.isError) {
    throw new Error(result.content[0].text as string);
  }

  const parsed = JSON.parse(result.content[0].text as string);
  const report = formatReviewReport(
    parsed.feedback as AggregatedFeedbackItem[],
    {
      dimensions: parsed.dimensions_reviewed,
      providers: parsed.providers_used as ProviderName[],
      budget,
    },
  );

  // Write output
  if (options.output) {
    deps.writeFileSync(options.output, report);
  }

  return report;
}

// ---------------------------------------------------------------------------
// CLI argument parser + entry point
// ---------------------------------------------------------------------------

function parseArgs(argv: string[]): { command: string; target: string; options: ReviewOptions } {
  const args = argv.slice(2); // skip node + script
  const command = args[0];
  const target = args[1];

  if (!command || !target) {
    console.error("Usage:");
    console.error("  node dist/cli.js review-pr <N> [--output path] [--post-comment] [--budget tier]");
    console.error("  node dist/cli.js review-plan <file> [--output path] [--budget tier]");
    process.exit(1);
  }

  const options: ReviewOptions = {};
  for (let i = 2; i < args.length; i++) {
    if (args[i] === "--output" && args[i + 1]) {
      options.output = args[++i];
    } else if (args[i] === "--post-comment") {
      options.postComment = true;
    } else if (args[i] === "--budget" && args[i + 1]) {
      options.budget = args[++i] as Budget;
    }
  }

  return { command, target, options };
}

async function main(): Promise<void> {
  const { command, target, options } = parseArgs(process.argv);
  const deps = createDefaultDeps();

  let report: string;
  if (command === "review-pr") {
    report = await reviewPR(target, options, deps);
  } else if (command === "review-plan") {
    report = await reviewPlan(target, options, deps);
  } else {
    console.error(`Unknown command: ${command}`);
    process.exit(1);
  }

  // Print to stdout if no output file specified
  if (!options.output) {
    process.stdout.write(report);
  }
}

// Run when executed directly
const isDirectRun = process.argv[1]?.endsWith("cli.js");
if (isDirectRun) {
  main().catch((err) => {
    console.error(`Error: ${(err as Error).message}`);
    process.exit(1);
  });
}
