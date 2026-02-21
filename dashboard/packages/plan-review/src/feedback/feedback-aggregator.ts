// ---------------------------------------------------------------------------
// Feedback aggregation — dedup (Jaccard) + rank (severity * corroboration)
// ---------------------------------------------------------------------------

import type {
  FeedbackItem,
  AggregatedFeedbackItem,
  Severity,
  ReviewDimension,
} from "../types.js";

const JACCARD_THRESHOLD = 0.45;

const SEVERITY_WEIGHT: Record<Severity, number> = {
  critical: 3.0,
  warning: 2.0,
  suggestion: 1.0,
};

const SEVERITY_LEVELS: Severity[] = ["suggestion", "warning", "critical"];

const DIMENSION_PRIORITY: Record<ReviewDimension, number> = {
  regression_risk: 5,
  blind_spots: 4,
  completeness: 3,
  test_coverage: 2,
  hypothesis_scope: 1,
};

function normalizeToWords(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 2),
  );
}

/**
 * Compute Jaccard similarity between two strings based on word sets.
 */
export function jaccardSimilarity(a: string, b: string): number {
  const setA = normalizeToWords(a);
  const setB = normalizeToWords(b);

  if (setA.size === 0 && setB.size === 0) return 1.0;
  if (setA.size === 0 || setB.size === 0) return 0.0;

  let intersection = 0;
  for (const word of setA) {
    if (setB.has(word)) intersection++;
  }

  const union = new Set([...setA, ...setB]).size;
  return intersection / union;
}

function promoteSeverity(base: Severity, extraSources: number): Severity {
  let idx = SEVERITY_LEVELS.indexOf(base);
  idx = Math.min(idx + extraSources, SEVERITY_LEVELS.length - 1);
  return SEVERITY_LEVELS[idx];
}

/**
 * Deduplicate and rank feedback items.
 *
 * Dedup rules:
 * - Jaccard similarity >= 0.45 AND same dimension = duplicate
 * - Merge: keep longest description, union affected_files, record all sources
 * - Severity promotion: +1 level per extra corroborating model (capped at critical)
 *
 * Ranking:
 * score = severity_weight + corroboration_bonus + file_specificity_bonus
 * Sort by score desc, then by dimension priority.
 */
export function aggregateFeedback(
  items: FeedbackItem[],
): AggregatedFeedbackItem[] {
  if (items.length === 0) return [];

  // Build clusters of similar items
  const clusters: Array<{
    items: FeedbackItem[];
    dimension: ReviewDimension;
  }> = [];

  for (const item of items) {
    let merged = false;
    for (const cluster of clusters) {
      if (cluster.dimension !== item.dimension) continue;

      // Check if any item in the cluster is similar
      const isSimilar = cluster.items.some(
        (existing) =>
          jaccardSimilarity(existing.description, item.description) >=
          JACCARD_THRESHOLD,
      );

      if (isSimilar) {
        cluster.items.push(item);
        merged = true;
        break;
      }
    }

    if (!merged) {
      clusters.push({ items: [item], dimension: item.dimension });
    }
  }

  // Convert clusters to aggregated items
  const aggregated: AggregatedFeedbackItem[] = clusters.map((cluster) => {
    const sources = [...new Set(cluster.items.map((i) => i.source))];
    const allFiles = [
      ...new Set(cluster.items.flatMap((i) => i.affected_files)),
    ];

    // Keep the longest description
    const longestItem = cluster.items.reduce((a, b) =>
      b.description.length > a.description.length ? b : a,
    );

    // Find highest base severity
    const baseSeverity = cluster.items.reduce((best, item) => {
      return SEVERITY_LEVELS.indexOf(item.severity) >
        SEVERITY_LEVELS.indexOf(best)
        ? item.severity
        : best;
    }, cluster.items[0].severity);

    // Promote severity based on number of extra corroborating sources
    const extraSources = Math.max(0, sources.length - 1);
    const severity = promoteSeverity(baseSeverity, extraSources);

    // Calculate score
    const corroborationBonus = extraSources * 1.0;
    const fileBonus = allFiles.length > 0 ? 0.5 : 0;
    const score = SEVERITY_WEIGHT[severity] + corroborationBonus + fileBonus;

    return {
      severity,
      description: longestItem.description,
      affected_files: allFiles,
      reasoning: longestItem.reasoning,
      dimension: cluster.dimension,
      corroborated_by: sources,
      score,
    };
  });

  // Sort by score desc, then dimension priority desc
  aggregated.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return (
      (DIMENSION_PRIORITY[b.dimension] ?? 0) -
      (DIMENSION_PRIORITY[a.dimension] ?? 0)
    );
  });

  return aggregated;
}
