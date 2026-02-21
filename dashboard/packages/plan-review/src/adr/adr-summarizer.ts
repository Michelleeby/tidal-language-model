// ---------------------------------------------------------------------------
// Deterministic ADR parser — no LLM calls, pure text extraction
// ---------------------------------------------------------------------------

import type { ADRSummary } from "../types.js";

/**
 * Extract the first paragraph after a markdown heading (## Section).
 * A paragraph ends at the next blank line or next heading.
 */
function extractSectionParagraph(content: string, sectionName: string): string {
  const heading = `## ${sectionName}`;
  const idx = content.indexOf(heading);
  if (idx === -1) return "";

  // Skip past the heading line and the blank line after it
  const afterHeading = content.slice(idx + heading.length);
  const bodyStart = afterHeading.indexOf("\n\n");
  if (bodyStart === -1) return "";

  const body = afterHeading.slice(bodyStart + 2);

  // Take content up to the next section heading (## ) or end
  const nextSection = body.indexOf("\n\n## ");
  const sectionBody = nextSection === -1 ? body : body.slice(0, nextSection);

  // Return only the first paragraph
  return sectionBody.trim().split("\n\n")[0].trim();
}

/**
 * Extract backtick-quoted identifiers (class names, paths, config keys).
 */
function extractBacktickKeywords(content: string): string[] {
  const matches = content.matchAll(/`([^`]+)`/g);
  const keywords = new Set<string>();
  for (const m of matches) {
    // Extract just the last path segment or class name for better matching
    const value = m[1];
    // If it's a path, extract the filename without extension
    const pathMatch = value.match(/([^/]+?)(?:\.\w+)?$/);
    if (pathMatch) {
      keywords.add(pathMatch[1]);
    }
    // Also add the full value for path matching
    keywords.add(value);
  }
  return [...keywords];
}

/**
 * Extract file paths from the References section.
 */
function extractFilesAffected(content: string): string[] {
  const files: string[] = [];

  // Match "Code:" and "Config:" references
  const refPattern = /- (?:Code|Config|File):\s*`([^`]+)`/g;
  for (const m of content.matchAll(refPattern)) {
    files.push(m[1]);
  }

  // Also check "Files affected" tables
  const tablePattern = /\|\s*`([^`]+)`\s*\|/g;
  for (const m of content.matchAll(tablePattern)) {
    if (m[1].includes("/")) {
      files.push(m[1]);
    }
  }

  return [...new Set(files)];
}

/**
 * Parse a single ADR markdown file into a structured summary.
 */
export function parseADR(content: string, filename: string): ADRSummary {
  // Number + title from "# {N}. {Title}"
  const headingMatch = content.match(/^# (\d+)\.\s+(.+)$/m);
  const number = headingMatch ? parseInt(headingMatch[1], 10) : 0;
  const title = headingMatch ? headingMatch[2].trim() : filename;

  // Status from "**Status:**" line
  const statusMatch = content.match(/\*\*Status:\*\*\s*(.+)/);
  const status = statusMatch ? statusMatch[1].trim() : "Unknown";

  // First paragraph of Context and Decision sections
  const context = extractSectionParagraph(content, "Context");
  const decision = extractSectionParagraph(content, "Decision");

  // Keywords and files
  const keywords = extractBacktickKeywords(content);
  const files_affected = extractFilesAffected(content);

  // Build compact summary string
  const raw = `ADR ${number}: ${title} (${status}). ${context} Decision: ${decision}`;

  return { number, title, status, context, decision, keywords, files_affected, raw };
}

/**
 * Parse multiple ADR files into sorted summaries.
 */
export function summarizeADRs(
  entries: Array<{ filename: string; content: string }>,
): ADRSummary[] {
  return entries
    .map((e) => parseADR(e.content, e.filename))
    .sort((a, b) => a.number - b.number);
}
