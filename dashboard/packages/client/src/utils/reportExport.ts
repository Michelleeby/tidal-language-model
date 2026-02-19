import type { ReportBlock } from "../components/reports/BlockEditor.js";

/**
 * Convert report blocks to a simplified Markdown string.
 * BlockNote stores rich inline content; we flatten to plain text.
 *
 * @param captures - Optional map of block ID → base64 data URL from canvas capture.
 *                   When provided, chart blocks emit inline images instead of placeholders.
 */
export function exportToMarkdown(
  title: string,
  blocks: ReportBlock[],
  captures?: Map<string, string>,
): string {
  const lines: string[] = [];

  lines.push(`# ${title}`, "");

  for (const block of blocks) {
    const text = flattenInlineContent(block);
    const capture = captures?.get(block.id);

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
      case "experimentChart": {
        const eProps = block.props as any;
        const eMode = eProps.chartMode || "lm";
        const eMetric =
          eMode === "rl" ? (eProps.rlMetricKey || "episode_rewards") :
          eMode === "ablation" ? (eProps.ablationMetricKey || "mean_reward") :
          (eProps.metricKey || "Losses/Total");
        const modeLabel = eMode === "rl" ? "RL" : eMode === "ablation" ? "Ablation" : "LM";
        const altText = `Chart (${modeLabel}): ${eProps.experimentId || "none"}`;
        if (capture) {
          lines.push(`![${altText}](${capture})`, "");
        } else {
          lines.push(
            `> **Chart** (${modeLabel}): Experiment \`${eProps.experimentId || "none"}\` — ${eMetric}`,
            "",
          );
        }
        break;
      }
      case "metricsTable":
        lines.push(
          `> **Metrics Table**: Experiment \`${(block.props as any).experimentId || "none"}\``,
          "",
        );
        break;
      case "trajectoryChart": {
        const tProps = block.props as any;
        const altText = `Gate Trajectory: ${tProps.experimentId || "none"}`;
        if (capture) {
          lines.push(`![${altText}](${capture})`, "");
        } else {
          lines.push(
            `> **Gate Trajectory**: Experiment \`${tProps.experimentId || "none"}\` — mode: ${tProps.gatingMode || "fixed"}, prompt: "${tProps.prompt || ""}"`,
            "",
          );
        }
        break;
      }
      case "crossPromptAnalysis": {
        const cpProps = block.props as any;
        const altText = `Cross-Prompt Analysis: ${cpProps.experimentId || "none"}`;
        if (capture) {
          lines.push(`![${altText}](${capture})`, "");
        } else {
          lines.push(
            `> **Cross-Prompt Analysis**: Experiment \`${cpProps.experimentId || "none"}\` — mode: ${cpProps.gatingMode || "fixed"}`,
            "",
          );
        }
        break;
      }
      case "sweepAnalysis": {
        const swProps = block.props as any;
        const altText = `Gate Sweep: ${swProps.experimentId || "none"}`;
        if (capture) {
          lines.push(`![${altText}](${capture})`, "");
        } else {
          lines.push(
            `> **Gate Sweep**: Experiment \`${swProps.experimentId || "none"}\` — prompt: "${swProps.prompt || ""}"`,
            "",
          );
        }
        break;
      }
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
      case "experimentChart": {
        const hProps = block.props as any;
        const hMode = hProps.chartMode || "lm";
        const hMetric =
          hMode === "rl" ? (hProps.rlMetricKey || "episode_rewards") :
          hMode === "ablation" ? (hProps.ablationMetricKey || "mean_reward") :
          (hProps.metricKey || "Losses/Total");
        const hModeLabel = hMode === "rl" ? "RL" : hMode === "ablation" ? "Ablation" : "LM";
        bodyParts.push(
          `<div class="chart-placeholder"><strong>Chart</strong> (${esc(hModeLabel)}): Experiment <code>${esc(hProps.experimentId || "none")}</code> &mdash; ${esc(hMetric)}</div>`,
        );
        break;
      }
      case "metricsTable":
        bodyParts.push(
          `<div class="metrics-placeholder"><strong>Metrics Table</strong>: Experiment <code>${esc((block.props as any).experimentId || "none")}</code></div>`,
        );
        break;
      case "trajectoryChart": {
        const htProps = block.props as any;
        bodyParts.push(
          `<div class="chart-placeholder"><strong>Gate Trajectory</strong>: Experiment <code>${esc(htProps.experimentId || "none")}</code> &mdash; mode: ${esc(htProps.gatingMode || "fixed")}, prompt: &ldquo;${esc(htProps.prompt || "")}&rdquo;</div>`,
        );
        break;
      }
      case "crossPromptAnalysis": {
        const hcpProps = block.props as any;
        bodyParts.push(
          `<div class="chart-placeholder"><strong>Cross-Prompt Analysis</strong>: Experiment <code>${esc(hcpProps.experimentId || "none")}</code> &mdash; mode: ${esc(hcpProps.gatingMode || "fixed")}</div>`,
        );
        break;
      }
      case "sweepAnalysis": {
        const hswProps = block.props as any;
        bodyParts.push(
          `<div class="chart-placeholder"><strong>Gate Sweep</strong>: Experiment <code>${esc(hswProps.experimentId || "none")}</code> &mdash; prompt: &ldquo;${esc(hswProps.prompt || "")}&rdquo;</div>`,
        );
        break;
      }
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

/**
 * Convert report blocks to an interactive HTML document with embedded data.
 * Custom chart blocks get their analysis data serialized as JSON <script> tags,
 * with inline JS that renders interactive charts using Canvas.
 *
 * @param analysisDataMap - Map of block ID → analysis data payload.
 */
export function exportToInteractiveHTML(
  title: string,
  blocks: ReportBlock[],
  analysisDataMap: Map<string, Record<string, unknown>>,
): string {
  const bodyParts: string[] = [];
  bodyParts.push(`<h1>${esc(title)}</h1>`);

  for (const block of blocks) {
    const text = esc(flattenInlineContent(block));
    const analysisData = analysisDataMap.get(block.id);

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
      case "experimentChart": {
        const eProps = block.props as any;
        const eMode = eProps.chartMode || "lm";
        const eMetric =
          eMode === "rl" ? (eProps.rlMetricKey || "episode_rewards") :
          eMode === "ablation" ? (eProps.ablationMetricKey || "mean_reward") :
          (eProps.metricKey || "Losses/Total");
        const eModeLabel = eMode === "rl" ? "RL" : eMode === "ablation" ? "Ablation" : "LM";
        if (analysisData) {
          const dataId = `data-${block.id}`;
          bodyParts.push(
            `<div class="chart-container">`,
            `  <div class="chart-label"><strong>Chart</strong> (${esc(eModeLabel)}): ${esc(eMetric)}</div>`,
            `  <script type="application/json" id="${dataId}">${JSON.stringify(analysisData).replace(/<\//g, "<\\/")}</script>`,
            `  <canvas id="canvas-${block.id}" width="600" height="200"></canvas>`,
            `</div>`,
          );
        } else {
          bodyParts.push(
            `<div class="chart-placeholder"><strong>Chart</strong> (${esc(eModeLabel)}): Experiment <code>${esc(eProps.experimentId || "none")}</code> &mdash; ${esc(eMetric)}</div>`,
          );
        }
        break;
      }
      case "metricsTable":
        bodyParts.push(
          `<div class="metrics-placeholder"><strong>Metrics Table</strong>: Experiment <code>${esc((block.props as any).experimentId || "none")}</code></div>`,
        );
        break;
      case "trajectoryChart": {
        const tProps = block.props as any;
        if (analysisData) {
          const dataId = `data-${block.id}`;
          bodyParts.push(
            `<div class="chart-container">`,
            `  <div class="chart-label"><strong>Gate Trajectory</strong>: ${esc(tProps.gatingMode || "fixed")} gating</div>`,
            `  <script type="application/json" id="${dataId}">${JSON.stringify(analysisData).replace(/<\//g, "<\\/")}</script>`,
            `  <canvas id="canvas-${block.id}" width="600" height="200"></canvas>`,
            `</div>`,
          );
        } else {
          bodyParts.push(
            `<div class="chart-placeholder"><strong>Gate Trajectory</strong>: Experiment <code>${esc(tProps.experimentId || "none")}</code> &mdash; mode: ${esc(tProps.gatingMode || "fixed")}</div>`,
          );
        }
        break;
      }
      case "crossPromptAnalysis": {
        const cpProps = block.props as any;
        if (analysisData) {
          const dataId = `data-${block.id}`;
          bodyParts.push(
            `<div class="chart-container">`,
            `  <div class="chart-label"><strong>Cross-Prompt Analysis</strong>: ${esc(cpProps.gatingMode || "fixed")} gating</div>`,
            `  <script type="application/json" id="${dataId}">${JSON.stringify(analysisData).replace(/<\//g, "<\\/")}</script>`,
            `  <canvas id="canvas-${block.id}" width="600" height="300"></canvas>`,
            `</div>`,
          );
        } else {
          bodyParts.push(
            `<div class="chart-placeholder"><strong>Cross-Prompt Analysis</strong>: Experiment <code>${esc(cpProps.experimentId || "none")}</code> &mdash; mode: ${esc(cpProps.gatingMode || "fixed")}</div>`,
          );
        }
        break;
      }
      case "sweepAnalysis": {
        const swProps = block.props as any;
        if (analysisData) {
          const dataId = `data-${block.id}`;
          bodyParts.push(
            `<div class="chart-container">`,
            `  <div class="chart-label"><strong>Gate Sweep</strong>: extreme-value analysis</div>`,
            `  <script type="application/json" id="${dataId}">${JSON.stringify(analysisData).replace(/<\//g, "<\\/")}</script>`,
            `  <canvas id="canvas-${block.id}" width="600" height="300"></canvas>`,
            `</div>`,
          );
        } else {
          bodyParts.push(
            `<div class="chart-placeholder"><strong>Gate Sweep</strong>: Experiment <code>${esc(swProps.experimentId || "none")}</code> &mdash; prompt: &ldquo;${esc(swProps.prompt || "")}&rdquo;</div>`,
          );
        }
        break;
      }
      default:
        if (text) bodyParts.push(`<p>${text}</p>`);
        break;
    }
  }

  // Inline chart rendering script
  const chartScript = `
<script>
(function() {
  // Render line chart on canvas from array of numbers
  function renderLineChart(canvasId, dataId, color) {
    var el = document.getElementById(dataId);
    if (!el) return;
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;
    var data = JSON.parse(el.textContent);
    var values = data.values || [];
    if (values.length === 0) return;
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    var min = Math.min.apply(null, values), max = Math.max.apply(null, values);
    var range = max - min || 1;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#374151'; ctx.lineWidth = 0.5;
    for (var i = 0; i < 4; i++) { var y = (H/4)*i; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }
    ctx.strokeStyle = color || '#3b82f6'; ctx.lineWidth = 1.5; ctx.beginPath();
    for (var j = 0; j < values.length; j++) {
      var x = values.length <= 1 ? W / 2 : (j / (values.length-1)) * W;
      var y = H - ((values[j]-min)/range) * (H-8) - 4;
      j === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.fillStyle = '#9ca3af'; ctx.font = '10px monospace';
    ctx.fillText(max.toFixed(4), 4, 12); ctx.fillText(min.toFixed(4), 4, H-4);
  }

  // Initialize all chart canvases
  document.querySelectorAll('.chart-container canvas').forEach(function(canvas) {
    var id = canvas.id.replace('canvas-', '');
    var dataEl = document.getElementById('data-' + id);
    if (!dataEl) return;
    var data = JSON.parse(dataEl.textContent);
    // Simple line chart for trajectory signals
    if (data.trajectory && data.trajectory.gateSignals) {
      var signals = data.trajectory.gateSignals;
      var ctx = canvas.getContext('2d');
      var W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = '#a78bfa'; ctx.lineWidth = 1.5; ctx.beginPath();
      for (var i = 0; i < signals.length; i++) {
        var x = signals.length <= 1 ? W / 2 : (i / (signals.length-1)) * W;
        var y = H - signals[i][0] * (H-8) - 4;
        i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
      }
      ctx.stroke();
      ctx.fillStyle = '#9ca3af'; ctx.font = '10px monospace';
      ctx.fillText('modulation', 4, 12);
    }
  });
})();
</script>`;

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
    .chart-container { background: #111827; border: 1px solid #374151; border-radius: 6px; padding: 1em; margin: 1em 0; }
    .chart-container canvas { width: 100%; background: #0a0a0f; border-radius: 4px; }
    .chart-label { font-size: 0.85em; color: #9ca3af; margin-bottom: 0.5em; }
    blockquote { border-left: 3px solid #3b82f6; padding-left: 1em; color: #9ca3af; }
  </style>
</head>
<body>
${bodyParts.join("\n")}
${chartScript}
</body>
</html>`;
}

// ---------------------------------------------------------------------------
// Canvas capture utility for markdown export
// ---------------------------------------------------------------------------

/**
 * Walk the editor DOM finding custom block containers and capture their
 * canvas elements as base64 PNG data URLs.
 *
 * @returns Map of block ID → base64 data URL
 */
export function captureBlockCanvases(
  editorContainer: HTMLElement,
): Map<string, string> {
  const captures = new Map<string, string>();

  // BlockNote wraps custom blocks in elements with data-content-type attribute.
  // The data-id attribute lives on an ancestor (.bn-block-outer), not on the
  // data-content-type element itself, so we use closest() to find it.
  const blockEls = editorContainer.querySelectorAll("[data-content-type]");
  for (const el of blockEls) {
    const blockId = el.closest("[data-id]")?.getAttribute("data-id") ?? "";
    if (!blockId) continue;

    const canvas = el.querySelector("canvas");
    if (!canvas) continue;

    try {
      const dataUrl = canvas.toDataURL("image/png");
      if (dataUrl && dataUrl.startsWith("data:image/png")) {
        captures.set(blockId, dataUrl);
      }
    } catch {
      // Canvas tainted or not available — skip
    }
  }

  return captures;
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
