// ---------------------------------------------------------------------------
// Markdown report formatter for review results
// ---------------------------------------------------------------------------

import type { AggregatedFeedbackItem, ProviderName, Budget } from "./types.js";

export interface ReportMeta {
  dimensions: string[];
  providers: ProviderName[];
  budget: Budget;
}

function formatItem(item: AggregatedFeedbackItem): string {
  const lines: string[] = [];
  lines.push(`- **${item.description}**`);
  if (item.affected_files.length > 0) {
    lines.push(`  - Files: ${item.affected_files.map((f) => `\`${f}\``).join(", ")}`);
  }
  lines.push(`  - ${item.reasoning}`);
  if (item.corroborated_by.length > 0) {
    lines.push(`  - Flagged by: ${item.corroborated_by.join(", ")}`);
  }
  return lines.join("\n");
}

export function formatReviewReport(
  feedback: AggregatedFeedbackItem[],
  meta: ReportMeta,
): string {
  const sections: string[] = [];

  sections.push("# Review Report\n");

  if (feedback.length === 0) {
    sections.push("No issues found. The review completed with no feedback items.\n");
    sections.push("## Summary\n");
    sections.push(`- **Dimensions reviewed:** ${meta.dimensions.join(", ")}`);
    sections.push(`- **Providers:** ${meta.providers.join(", ")}`);
    sections.push(`- **Budget:** ${meta.budget}`);
    return sections.join("\n");
  }

  const critical = feedback.filter((f) => f.severity === "critical");
  const warnings = feedback.filter((f) => f.severity === "warning");
  const suggestions = feedback.filter((f) => f.severity === "suggestion");

  if (critical.length > 0) {
    sections.push("## Critical Issues\n");
    sections.push(critical.map(formatItem).join("\n\n"));
    sections.push("");
  }

  if (warnings.length > 0) {
    sections.push("## Warnings\n");
    sections.push(warnings.map(formatItem).join("\n\n"));
    sections.push("");
  }

  if (suggestions.length > 0) {
    sections.push("## Suggestions\n");
    sections.push(suggestions.map(formatItem).join("\n\n"));
    sections.push("");
  }

  // Summary
  const dimCounts: Record<string, number> = {};
  for (const item of feedback) {
    dimCounts[item.dimension] = (dimCounts[item.dimension] ?? 0) + 1;
  }

  sections.push("## Summary\n");
  sections.push(`- **Total issues:** ${feedback.length} (${critical.length} critical, ${warnings.length} warnings, ${suggestions.length} suggestions)`);

  const dimLines = Object.entries(dimCounts)
    .map(([dim, count]) => `${dim}: ${count}`)
    .join(", ");
  sections.push(`- **By dimension:** ${dimLines}`);
  sections.push(`- **Providers:** ${meta.providers.join(", ")}`);
  sections.push(`- **Budget:** ${meta.budget}`);

  if (critical.length > 0) {
    sections.push("\n**Assessment:** Significant gaps identified — address critical issues before proceeding.");
  } else if (warnings.length > 0) {
    sections.push("\n**Assessment:** Some concerns to address.");
  } else {
    sections.push("\n**Assessment:** Plan looks solid.");
  }

  return sections.join("\n");
}
