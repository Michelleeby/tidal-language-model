import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { ProviderRegistry } from "../providers/provider.js";
import type { LLMProvider, ReviewRequest, ReviewResponse } from "../providers/provider.js";
import type { ProviderName, FeedbackItem, Budget } from "../types.js";
import { reviewPR, reviewPlan, type CLIDeps } from "../cli.js";

// ---------------------------------------------------------------------------
// Mock helpers
// ---------------------------------------------------------------------------

function createMockProvider(name: ProviderName): LLMProvider {
  return {
    name,
    available: true,
    models: name === "openai" ? ["gpt-4o", "o3-mini"] : ["gemini-2.0-flash"],
    async review(request: ReviewRequest, model?: string): Promise<ReviewResponse> {
      return {
        feedback: [
          {
            severity: "warning",
            description: `${name} found a potential issue`,
            affected_files: ["src/main.ts"],
            reasoning: `${name} analysis`,
            dimension: request.dimension,
            source: name,
          },
        ],
        provider: name,
        model: model ?? "mock-model",
        dimension: request.dimension,
      };
    },
    estimateTokens: (t) => Math.ceil(t.length / 4),
  };
}

function makeMockDeps(overrides?: Partial<CLIDeps>): CLIDeps {
  const registry = new ProviderRegistry();
  registry.register(createMockProvider("openai"));
  registry.register(createMockProvider("google"));

  const writtenFiles: Record<string, string> = {};

  return {
    execGH: (args: string[]) => {
      if (args[0] === "pr" && args[1] === "diff") {
        return "diff --git a/src/main.ts b/src/main.ts\n+some change";
      }
      if (args[0] === "pr" && args[1] === "view") {
        return JSON.stringify({ title: "Test PR", body: "PR body" });
      }
      if (args[0] === "pr" && args[1] === "comment") {
        return "comment posted";
      }
      return "";
    },
    readFileSync: (_path: string) => "# Test Plan\n\nSome plan content",
    writeFileSync: (path: string, content: string) => {
      writtenFiles[path] = content;
    },
    writtenFiles,
    registry,
    adrDir: null,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// reviewPR
// ---------------------------------------------------------------------------

describe("reviewPR", () => {
  it("exits with error if gh is not available", async () => {
    const deps = makeMockDeps({
      execGH: () => { throw new Error("gh: command not found"); },
    });

    await assert.rejects(
      () => reviewPR("28", { budget: "minimal" }, deps),
      (err: Error) => err.message.includes("gh"),
    );
  });

  it("calls review pipeline with correct diff", async () => {
    const diff = "diff --git a/file.ts b/file.ts\n+new code";
    const deps = makeMockDeps({
      execGH: (args: string[]) => {
        if (args[0] === "pr" && args[1] === "diff") return diff;
        if (args[0] === "pr" && args[1] === "view") {
          return JSON.stringify({ title: "Test", body: "" });
        }
        return "";
      },
    });

    const report = await reviewPR("28", { budget: "minimal" }, deps);
    assert.ok(report.length > 0, "should produce a non-empty report");
    assert.ok(report.includes("##"), "should contain markdown headings");
  });

  it("writes to output file when --output is specified", async () => {
    const deps = makeMockDeps();
    await reviewPR("28", { output: "/tmp/review.md", budget: "minimal" }, deps);

    assert.ok(
      deps.writtenFiles["/tmp/review.md"],
      "should write report to output path",
    );
    assert.ok(
      deps.writtenFiles["/tmp/review.md"].includes("##"),
      "written content should be markdown",
    );
  });

  it("posts comment when --post-comment is specified", async () => {
    let commentPosted = false;
    const deps = makeMockDeps({
      execGH: (args: string[]) => {
        if (args[0] === "pr" && args[1] === "comment") {
          commentPosted = true;
          return "ok";
        }
        if (args[0] === "pr" && args[1] === "diff") {
          return "diff --git a/f.ts b/f.ts\n+x";
        }
        if (args[0] === "pr" && args[1] === "view") {
          return JSON.stringify({ title: "T", body: "" });
        }
        return "";
      },
    });

    await reviewPR("28", { postComment: true, budget: "minimal" }, deps);
    assert.ok(commentPosted, "should post PR comment");
  });
});

// ---------------------------------------------------------------------------
// reviewPlan
// ---------------------------------------------------------------------------

describe("reviewPlan", () => {
  it("reads file and produces review report", async () => {
    const planContent = "# Plan\n\nAdd new feature for gating controller";
    const deps = makeMockDeps({
      readFileSync: () => planContent,
    });

    const report = await reviewPlan("plan.md", { budget: "minimal" }, deps);
    assert.ok(report.length > 0, "should produce a non-empty report");
    assert.ok(report.includes("##"), "should contain markdown headings");
  });

  it("writes to output file when specified", async () => {
    const deps = makeMockDeps();
    await reviewPlan("plan.md", { output: "/tmp/plan-review.md", budget: "minimal" }, deps);

    assert.ok(
      deps.writtenFiles["/tmp/plan-review.md"],
      "should write report to output path",
    );
  });

  it("throws if plan file cannot be read", async () => {
    const deps = makeMockDeps({
      readFileSync: () => { throw new Error("ENOENT: no such file"); },
    });

    await assert.rejects(
      () => reviewPlan("nonexistent.md", { budget: "minimal" }, deps),
      (err: Error) => err.message.includes("ENOENT") || err.message.includes("read"),
    );
  });
});
