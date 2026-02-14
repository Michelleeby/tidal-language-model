import type { ReportBlock } from "../components/reports/BlockEditor.js";

/**
 * Convert report blocks to a simplified Markdown string.
 * BlockNote stores rich inline content; we flatten to plain text.
 */
export function exportToMarkdown(title: string, blocks: ReportBlock[]): string {
  const lines: string[] = [];

  lines.push(`# ${title}`, "");

  for (const block of blocks) {
    const text = flattenInlineContent(block);

    switch (block.type) {
      case "heading": {
        const level = (block.props as any).level ?? 1;
        lines.push(`${"#".repeat(level + 1)} ${text}`, "");
        break;
      }
      case "bulletListItem":
        lines.push(`- ${text}`);
        break;
      case "numberedListItem":
        lines.push(`1. ${text}`);
        break;
      case "checkListItem":
        lines.push(
          `- [${(block.props as any).checked ? "x" : " "}] ${text}`,
        );
        break;
      case "codeBlock":
        lines.push(`\`\`\``, text, `\`\`\``, "");
        break;
      case "table":
        lines.push(renderTableMarkdown(block), "");
        break;
      case "experimentChart":
        lines.push(
          `> **Chart**: Experiment \`${(block.props as any).experimentId || "none"}\` — ${(block.props as any).metricKey || "Losses/Total"}`,
          "",
        );
        break;
      case "metricsTable":
        lines.push(
          `> **Metrics Table**: Experiment \`${(block.props as any).experimentId || "none"}\``,
          "",
        );
        break;
      default:
        if (text) lines.push(text, "");
        else lines.push("");
        break;
    }
  }

  return lines.join("\n");
}

/**
 * Convert report blocks to a standalone HTML document.
 */
export function exportToHTML(title: string, blocks: ReportBlock[]): string {
  const bodyParts: string[] = [];

  bodyParts.push(`<h1>${esc(title)}</h1>`);

  for (const block of blocks) {
    const text = esc(flattenInlineContent(block));

    switch (block.type) {
      case "heading": {
        const level = Math.min(((block.props as any).level ?? 1) + 1, 6);
        bodyParts.push(`<h${level}>${text}</h${level}>`);
        break;
      }
      case "bulletListItem":
        bodyParts.push(`<ul><li>${text}</li></ul>`);
        break;
      case "numberedListItem":
        bodyParts.push(`<ol><li>${text}</li></ol>`);
        break;
      case "checkListItem":
        bodyParts.push(
          `<ul><li><input type="checkbox" ${(block.props as any).checked ? "checked" : ""} disabled> ${text}</li></ul>`,
        );
        break;
      case "codeBlock":
        bodyParts.push(`<pre><code>${text}</code></pre>`);
        break;
      case "table":
        bodyParts.push(renderTableHTML(block));
        break;
      case "experimentChart":
        bodyParts.push(
          `<div class="chart-placeholder"><strong>Chart</strong>: Experiment <code>${esc((block.props as any).experimentId || "none")}</code> &mdash; ${esc((block.props as any).metricKey || "Losses/Total")}</div>`,
        );
        break;
      case "metricsTable":
        bodyParts.push(
          `<div class="metrics-placeholder"><strong>Metrics Table</strong>: Experiment <code>${esc((block.props as any).experimentId || "none")}</code></div>`,
        );
        break;
      default:
        if (text) bodyParts.push(`<p>${text}</p>`);
        break;
    }
  }

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${esc(title)}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; color: #e5e7eb; background: #030712; }
    h1, h2, h3, h4 { color: #f3f4f6; }
    code { background: #1f2937; padding: 0.15em 0.4em; border-radius: 3px; font-size: 0.9em; }
    pre { background: #111827; padding: 1em; border-radius: 6px; overflow-x: auto; }
    pre code { background: none; padding: 0; }
    table { border-collapse: collapse; width: 100%; margin: 1em 0; }
    th, td { border: 1px solid #374151; padding: 0.5em 0.75em; text-align: left; }
    th { background: #1f2937; }
    .chart-placeholder, .metrics-placeholder { background: #1f2937; border: 1px solid #374151; border-radius: 6px; padding: 1em; margin: 1em 0; }
    blockquote { border-left: 3px solid #3b82f6; padding-left: 1em; color: #9ca3af; }
  </style>
</head>
<body>
${bodyParts.join("\n")}
</body>
</html>`;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function flattenInlineContent(block: any): string {
  const content = block.content;
  if (!content) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((c: any) => {
        if (typeof c === "string") return c;
        if (c.type === "text") return c.text ?? "";
        if (c.type === "link") {
          const linkText = (c.content ?? [])
            .map((t: any) => t.text ?? "")
            .join("");
          return linkText;
        }
        return c.text ?? "";
      })
      .join("");
  }
  return "";
}

function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderTableMarkdown(block: any): string {
  const rows: any[] = block.content?.rows ?? block.children ?? [];
  if (rows.length === 0) return "";

  const lines: string[] = [];
  for (let i = 0; i < rows.length; i++) {
    const cells = rows[i].cells ?? [];
    const rowText = cells.map((cell: any) => {
      if (Array.isArray(cell)) {
        return cell.map((c: any) => c.text ?? "").join("");
      }
      return String(cell);
    });
    lines.push(`| ${rowText.join(" | ")} |`);
    if (i === 0) {
      lines.push(`| ${rowText.map(() => "---").join(" | ")} |`);
    }
  }
  return lines.join("\n");
}

function renderTableHTML(block: any): string {
  const rows: any[] = block.content?.rows ?? block.children ?? [];
  if (rows.length === 0) return "";

  let html = "<table>";
  for (let i = 0; i < rows.length; i++) {
    const cells = rows[i].cells ?? [];
    const tag = i === 0 ? "th" : "td";
    const wrapper = i === 0 ? "thead" : "tbody";
    if (i <= 1) html += `<${wrapper}>`;
    html += "<tr>";
    for (const cell of cells) {
      const text = Array.isArray(cell)
        ? cell.map((c: any) => esc(c.text ?? "")).join("")
        : esc(String(cell));
      html += `<${tag}>${text}</${tag}>`;
    }
    html += "</tr>";
    if (i === 0) html += "</thead>";
    if (i === rows.length - 1 && i > 0) html += "</tbody>";
  }
  html += "</table>";
  return html;
}
