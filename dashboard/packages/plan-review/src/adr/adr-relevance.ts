// ---------------------------------------------------------------------------
// ADR relevance scoring — keyword + file overlap with plan text
// ---------------------------------------------------------------------------

import type { ADRSummary } from "../types.js";

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function normalizeWords(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^\w\s/.-]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 2),
  );
}

/**
 * Score an ADR's relevance to the plan text.
 * Higher = more relevant.
 */
function scoreADR(adr: ADRSummary, planWords: Set<string>, planText: string): number {
  let score = 0;

  // Keyword overlap: each matching keyword adds 2 points
  for (const kw of adr.keywords) {
    const kwLower = kw.toLowerCase();
    if (planWords.has(kwLower) || planText.toLowerCase().includes(kwLower)) {
      score += 2;
    }
  }

  // File overlap: each matching file adds 5 points (strong signal)
  for (const file of adr.files_affected) {
    if (planText.includes(file)) {
      score += 5;
    }
  }

  return score;
}

/**
 * Select the most relevant ADRs for a given plan, respecting a token budget.
 *
 * Always includes:
 * - ADRs whose files_affected overlap with files mentioned in the plan
 * - The most recent ADR (highest number)
 *
 * Then greedily adds by score until the token budget is exhausted.
 */
export function selectRelevantADRs(
  summaries: ADRSummary[],
  planText: string,
  tokenBudget: number,
): ADRSummary[] {
  if (summaries.length === 0) return [];

  const planWords = normalizeWords(planText);

  // Score all ADRs
  const scored = summaries.map((adr) => ({
    adr,
    score: scoreADR(adr, planWords, planText),
  }));

  // Sort by score descending, then by number descending (recent first for ties)
  scored.sort((a, b) => b.score - a.score || b.adr.number - a.adr.number);

  const selected = new Map<number, ADRSummary>();
  let tokensUsed = 0;

  // Always include most recent ADR
  const mostRecent = summaries.reduce((a, b) => (a.number > b.number ? a : b));
  const mostRecentTokens = estimateTokens(mostRecent.raw);
  if (mostRecentTokens <= tokenBudget) {
    selected.set(mostRecent.number, mostRecent);
    tokensUsed += mostRecentTokens;
  }

  // Greedily add by score
  for (const { adr } of scored) {
    if (selected.has(adr.number)) continue;

    const tokens = estimateTokens(adr.raw);
    if (tokensUsed + tokens > tokenBudget) continue;

    selected.set(adr.number, adr);
    tokensUsed += tokens;
  }

  // Sort result by ADR number for consistent output
  return [...selected.values()].sort((a, b) => a.number - b.number);
}
