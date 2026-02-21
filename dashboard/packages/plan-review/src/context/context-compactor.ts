// ---------------------------------------------------------------------------
// Context compaction — fit plan + ADRs + code into model token budgets
// ---------------------------------------------------------------------------

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

interface CompactInput {
  plan: string;
  adrSummaries: string;
  codeContext: string;
  tokenBudget: number;
}

interface CompactResult extends String {
  truncated: boolean;
}

function makeResult(text: string, truncated: boolean): CompactResult {
  const result = new String(text) as CompactResult;
  result.truncated = truncated;
  return result;
}

/** Strip extra blank lines (3+ newlines → 2). */
function stripExtraBlankLines(text: string): string {
  return text.replace(/\n{3,}/g, "\n\n");
}

/** Remove horizontal rules. */
function stripHorizontalRules(text: string): string {
  return text.replace(/^-{3,}$/gm, "").replace(/^\*{3,}$/gm, "");
}

/** Collapse code blocks to just signatures (first line of each block). */
function collapseCodeBlocks(text: string): string {
  return text.replace(
    /```(\w*)\n([\s\S]*?)```/g,
    (_match, lang: string, body: string) => {
      const lines = body.trim().split("\n");
      // Keep lines that look like signatures (def, class, function, export, etc.)
      const signatures = lines.filter(
        (l) =>
          /^\s*(def |class |function |export |import |interface |type |const |let |var )/.test(l) ||
          lines.indexOf(l) === 0,
      );
      const collapsed = signatures.length > 0 ? signatures.join("\n") : lines[0] ?? "";
      return `\`\`\`${lang}\n${collapsed}\n\`\`\``;
    },
  );
}

/** Drop "Alternatives Considered" sections from ADR summaries. */
function dropAlternatives(text: string): string {
  return text.replace(
    /## Alternatives Considered[\s\S]*?(?=\n## [A-Z]|$)/g,
    "",
  );
}

/** Collapse ADR references to filenames only. */
function collapseReferences(text: string): string {
  return text.replace(
    /## References\n([\s\S]*?)(?=\n##|\n\n\n|$)/g,
    (_match, body: string) => {
      const files = body
        .split("\n")
        .map((l) => {
          const m = l.match(/`([^`]+)`/);
          return m ? `- ${m[1]}` : null;
        })
        .filter(Boolean);
      return files.length > 0 ? `## References\n${files.join("\n")}` : "";
    },
  );
}

/**
 * Compact plan + ADR summaries + code context to fit within a token budget.
 *
 * Priority order (highest first):
 * 1. Plan text
 * 2. ADR summaries
 * 3. Code context
 *
 * Compaction stages (applied progressively):
 * 1. Strip markdown noise (extra blank lines, horizontal rules)
 * 2. Collapse code blocks to signatures
 * 3. Drop ADR "Alternatives Considered" sections
 * 4. Collapse ADR references to filenames only
 * 5. Trim code context
 * 6. Truncate plan from end with [truncated] marker
 */
export function compactContext(input: CompactInput): CompactResult {
  let plan = input.plan;
  let adrs = input.adrSummaries;
  let code = input.codeContext;
  let truncated = false;

  // Stage 1: Strip markdown noise from all inputs
  plan = stripHorizontalRules(stripExtraBlankLines(plan));
  adrs = stripHorizontalRules(stripExtraBlankLines(adrs));
  code = stripHorizontalRules(stripExtraBlankLines(code));

  const assemble = () => {
    const parts = [plan, adrs, code].filter(Boolean);
    return parts.join("\n\n");
  };

  if (estimateTokens(assemble()) <= input.tokenBudget) {
    return makeResult(assemble(), false);
  }

  // Stage 2: Collapse code blocks to signatures
  code = collapseCodeBlocks(code);
  plan = collapseCodeBlocks(plan);

  if (estimateTokens(assemble()) <= input.tokenBudget) {
    return makeResult(assemble(), false);
  }

  // Stage 3: Drop ADR alternatives
  adrs = dropAlternatives(adrs);

  if (estimateTokens(assemble()) <= input.tokenBudget) {
    return makeResult(assemble(), false);
  }

  // Stage 4: Collapse ADR references
  adrs = collapseReferences(adrs);

  if (estimateTokens(assemble()) <= input.tokenBudget) {
    return makeResult(assemble(), false);
  }

  // Stage 5: Trim code context before plan
  const planTokens = estimateTokens(plan);
  const adrTokens = estimateTokens(adrs);
  const remainingForCode = input.tokenBudget - planTokens - adrTokens;
  if (remainingForCode <= 0) {
    code = "";
  } else {
    const maxChars = remainingForCode * 4;
    if (code.length > maxChars) {
      code = code.slice(0, maxChars) + "\n[code context truncated]";
    }
  }

  if (estimateTokens(assemble()) <= input.tokenBudget) {
    return makeResult(assemble(), false);
  }

  // Stage 6: Trim ADRs
  const remainingForAdrs = input.tokenBudget - planTokens;
  if (remainingForAdrs <= 0) {
    adrs = "";
  } else {
    const maxChars = remainingForAdrs * 4;
    if (adrs.length > maxChars) {
      adrs = adrs.slice(0, maxChars);
    }
  }

  if (estimateTokens(assemble()) <= input.tokenBudget) {
    return makeResult(assemble(), false);
  }

  // Stage 7: Truncate plan (last resort)
  truncated = true;
  const maxPlanChars = input.tokenBudget * 4 - 20; // room for marker
  if (maxPlanChars > 0) {
    plan = plan.slice(0, maxPlanChars) + "\n[truncated]";
  } else {
    plan = "[truncated]";
  }
  adrs = "";
  code = "";

  return makeResult(assemble(), truncated);
}
