/**
 * Shared chart data extraction and canvas rendering functions.
 *
 * Data extraction functions are pure (no DOM/Canvas) — testable in any environment.
 * Canvas rendering functions accept an HTMLCanvasElement and draw directly,
 * used by report block components, the analysis section, and HTML export.
 */

import type {
  BatchAnalysis,
  SweepAnalysis,
  RLTrainingHistory,
  AblationResults,
} from "@tidal/shared";

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

export const SIGNAL_NAMES = ["modulation"] as const;
export const SIGNAL_COLORS = ["#a78bfa"] as const;
export const HEATMAP_DIMENSIONS = { width: 600, height: 300 } as const;
export const SWEEP_DIMENSIONS = { width: 600, rowHeight: 80, headerHeight: 24 } as const;
export const TEXT_PROP_LABELS: Record<string, string> = {
  wordCount: "Words",
  uniqueTokenRatio: "Unique Ratio",
  charCount: "Chars",
};

const TEXT_PROPS = ["wordCount", "uniqueTokenRatio", "charCount"] as const;

// ---------------------------------------------------------------------------
// LM mini chart
// ---------------------------------------------------------------------------

/** Extract numeric values for a metric key, handling derived metrics. */
export function extractMiniChartData(
  points: Array<Record<string, unknown>>,
  metricKey: string,
): number[] {
  if (metricKey === "Perplexity") {
    return points
      .map((p) => Math.exp(Number(p["Losses/Total"])))
      .filter((v) => !isNaN(v) && isFinite(v));
  }
  return points
    .map((p) => Number(p[metricKey]))
    .filter((v) => !isNaN(v));
}

// ---------------------------------------------------------------------------
// RL mini chart
// ---------------------------------------------------------------------------

const RL_KEYS: ReadonlySet<string> = new Set([
  "episode_rewards",
  "policy_loss",
  "value_loss",
  "entropy",
  "gate_modulation",
  "reward_perplexity",
  "reward_diversity",
  "reward_sampling",
  "reward_repetition",
  "reward_coherence",
  "explained_variance",
]);

/** Extract a numeric array from an RLTrainingHistory by key. */
export function extractMiniRLChartData(
  history: RLTrainingHistory | null,
  key: string,
): number[] {
  if (!history || !RL_KEYS.has(key)) return [];
  return (
    (history[key as keyof RLTrainingHistory] as number[] | undefined) ?? []
  ).filter((v) => !isNaN(v));
}

// ---------------------------------------------------------------------------
// Ablation grouped bar chart
// ---------------------------------------------------------------------------

const ABLATION_SERIES = [
  { key: "mean_reward", label: "reward", color: "#3b82f6" },
  { key: "mean_diversity", label: "diversity", color: "#22c55e" },
  { key: "mean_perplexity", label: "perplexity", color: "#a855f7" },
] as const;

/** Extract all ablation metrics grouped by policy for a grouped bar chart. */
export function extractMiniAblationChartData(
  results: AblationResults | null,
): {
  policies: string[];
  series: Array<{
    key: string;
    label: string;
    color: string;
    values: number[];
  }>;
} {
  if (!results) return { policies: [], series: [] };
  const policies = Object.keys(results);
  const series = ABLATION_SERIES.map((m) => ({
    ...m,
    values: policies.map((p) => (results[p] as any)[m.key] as number),
  }));
  return { policies, series };
}

// ---------------------------------------------------------------------------
// Heatmap (cross-prompt analysis) — data extraction
// ---------------------------------------------------------------------------

export interface HeatmapRenderData {
  prompts: string[];
  signals: string[];
  values: number[][];
}

/** Extract heatmap data from batch analysis. */
export function extractHeatmapRenderData(batch: BatchAnalysis): HeatmapRenderData {
  const summaries = batch.perPromptSummaries as Record<string, any>;
  const prompts = Object.keys(summaries);
  const signals = [...SIGNAL_NAMES];
  const values = prompts.map((p) =>
    signals.map((s) => summaries[p]?.signalStats?.[s]?.mean ?? 0),
  );
  return { prompts, signals, values };
}

/** Extract variance summary per signal. */
export function extractVarianceSummary(
  crossPromptVariance: Record<string, { betweenPromptVar: number; withinPromptVar: number }>,
): Array<{ signal: string; between: number; within: number }> {
  return Object.entries(crossPromptVariance).map(([signal, v]) => ({
    signal,
    between: v.betweenPromptVar,
    within: v.withinPromptVar,
  }));
}

// ---------------------------------------------------------------------------
// Sweep panel — data extraction
// ---------------------------------------------------------------------------

export interface SweepPropertyRenderData {
  name: string;
  low: number;
  high: number;
  delta: number;
}

export interface SweepSignalRenderData {
  signal: string;
  properties: SweepPropertyRenderData[];
}

/** Extract per-signal low/high/delta for each text property. */
export function extractSweepRenderData(
  sweep: SweepAnalysis,
): SweepSignalRenderData[] {
  const imap = sweep.interpretabilityMap;
  if (!imap || Object.keys(imap).length === 0) return [];

  return Object.entries(imap).map(([signal, entry]) => ({
    signal,
    properties: TEXT_PROPS.filter((prop) => entry.effect[prop]).map((prop) => ({
      name: prop,
      low: entry.effect[prop].low,
      high: entry.effect[prop].high,
      delta: entry.effect[prop].delta,
    })),
  }));
}

/** Generate human-readable summary strings from the interpretability map. */
export function extractInterpretabilitySummary(
  interpretabilityMap: SweepAnalysis["interpretabilityMap"],
): string[] {
  if (!interpretabilityMap || Object.keys(interpretabilityMap).length === 0) {
    return [];
  }

  return Object.entries(interpretabilityMap).map(([signal, entry]) => {
    const parts = TEXT_PROPS
      .filter((prop) => entry.effect[prop])
      .map((prop) => {
        const d = entry.effect[prop].delta;
        const sign = d >= 0 ? "+" : "";
        const label = TEXT_PROP_LABELS[prop] ?? prop;
        return `${label}: ${sign}${typeof d === "number" && d % 1 !== 0 ? d.toFixed(2) : d}`;
      });
    return `${signal} low→high: ${parts.join(", ")}`;
  });
}

// ---------------------------------------------------------------------------
// Heatmap — canvas rendering
// ---------------------------------------------------------------------------

/**
 * Render a heatmap of signal means per prompt onto a canvas.
 * Safe to call with null canvas (no-op).
 */
export function renderHeatmap(
  canvas: HTMLCanvasElement | null,
  data: HeatmapRenderData,
): void {
  if (!canvas || data.prompts.length === 0) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const W = HEATMAP_DIMENSIONS.width;
  const H = HEATMAP_DIMENSIONS.height;

  const dpr = typeof window !== "undefined" ? (window.devicePixelRatio || 1) : 1;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = `${W}px`;
  canvas.style.height = `${H}px`;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const labelW = 180;
  const headerH = 24;
  const cellW = (W - labelW) / data.signals.length;
  const cellH = Math.min(24, (H - headerH) / data.prompts.length);

  // Column headers
  ctx.font = "11px monospace";
  ctx.textAlign = "center";
  for (let j = 0; j < data.signals.length; j++) {
    const colorIdx = Math.min(j, SIGNAL_COLORS.length - 1);
    ctx.fillStyle = SIGNAL_COLORS[colorIdx];
    ctx.fillText(
      data.signals[j],
      labelW + j * cellW + cellW / 2,
      headerH - 6,
    );
  }

  // Rows
  for (let i = 0; i < data.prompts.length; i++) {
    const y = headerH + i * cellH;

    // Prompt label (truncated)
    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    const label =
      data.prompts[i].length > 28
        ? data.prompts[i].slice(0, 28) + "..."
        : data.prompts[i];
    ctx.fillText(label, labelW - 8, y + cellH / 2 + 3);

    // Cells
    for (let j = 0; j < data.signals.length; j++) {
      const v = data.values[i][j]; // [0,1]
      const x = labelW + j * cellW;
      const colorIdx = Math.min(j, SIGNAL_COLORS.length - 1);

      // Color: blend from dark to signal color based on value
      const r = parseInt(SIGNAL_COLORS[colorIdx].slice(1, 3), 16);
      const g = parseInt(SIGNAL_COLORS[colorIdx].slice(3, 5), 16);
      const b = parseInt(SIGNAL_COLORS[colorIdx].slice(5, 7), 16);
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.15 + v * 0.85})`;
      ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);

      // Value text
      ctx.fillStyle = v > 0.5 ? "#111827" : "#e5e7eb";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(v.toFixed(2), x + cellW / 2, y + cellH / 2 + 3);
    }
  }
}

// ---------------------------------------------------------------------------
// Sweep panel — canvas rendering
// ---------------------------------------------------------------------------

/**
 * Render a sweep low/high comparison panel onto a canvas.
 * Safe to call with null canvas (no-op).
 */
export function renderSweepPanel(
  canvas: HTMLCanvasElement | null,
  data: SweepSignalRenderData[],
): void {
  if (!canvas || data.length === 0) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const PANEL_W = SWEEP_DIMENSIONS.width;
  const ROW_H = SWEEP_DIMENSIONS.rowHeight;
  const HEADER_H = SWEEP_DIMENSIONS.headerHeight;
  const totalH = HEADER_H + data.length * ROW_H;

  const dpr = typeof window !== "undefined" ? (window.devicePixelRatio || 1) : 1;
  canvas.width = PANEL_W * dpr;
  canvas.height = totalH * dpr;
  canvas.style.width = `${PANEL_W}px`;
  canvas.style.height = `${totalH}px`;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, PANEL_W, totalH);

  const labelW = 80;
  const chartW = PANEL_W - labelW - 20;
  const barH = 16;
  const barGap = 4;

  // Header
  ctx.font = "11px monospace";
  ctx.fillStyle = "#9ca3af";
  ctx.textAlign = "center";
  const propW = chartW / TEXT_PROPS.length;
  for (let j = 0; j < TEXT_PROPS.length; j++) {
    ctx.fillText(
      TEXT_PROP_LABELS[TEXT_PROPS[j]] ?? TEXT_PROPS[j],
      labelW + j * propW + propW / 2,
      HEADER_H - 6,
    );
  }

  for (let i = 0; i < data.length; i++) {
    const y = HEADER_H + i * ROW_H;
    const signalIdx = SIGNAL_NAMES.indexOf(data[i].signal as any);
    const color = SIGNAL_COLORS[signalIdx >= 0 ? signalIdx : 0];

    // Signal label
    ctx.fillStyle = color;
    ctx.font = "11px monospace";
    ctx.textAlign = "right";
    ctx.fillText(data[i].signal, labelW - 8, y + ROW_H / 2 + 4);

    for (let j = 0; j < data[i].properties.length; j++) {
      const prop = data[i].properties[j];
      const px = labelW + j * propW;
      const barY = y + 8;

      // Find max absolute value for scaling
      const maxVal = Math.max(Math.abs(prop.low), Math.abs(prop.high), 1);

      // Low bar
      const lowW = (Math.abs(prop.low) / maxVal) * (propW / 2 - 8);
      ctx.fillStyle = `${color}66`;
      ctx.fillRect(px + 4, barY, lowW, barH);
      ctx.fillStyle = "#9ca3af";
      ctx.font = "9px monospace";
      ctx.textAlign = "left";
      ctx.fillText(
        typeof prop.low === "number" && prop.low % 1 !== 0
          ? prop.low.toFixed(2)
          : String(prop.low),
        px + 4 + lowW + 2,
        barY + barH - 3,
      );

      // High bar
      const highY = barY + barH + barGap;
      const highW = (Math.abs(prop.high) / maxVal) * (propW / 2 - 8);
      ctx.fillStyle = color;
      ctx.fillRect(px + 4, highY, highW, barH);
      ctx.fillStyle = "#e5e7eb";
      ctx.font = "9px monospace";
      ctx.textAlign = "left";
      ctx.fillText(
        typeof prop.high === "number" && prop.high % 1 !== 0
          ? prop.high.toFixed(2)
          : String(prop.high),
        px + 4 + highW + 2,
        highY + barH - 3,
      );

      // Delta annotation
      const deltaY = highY + barH + 12;
      const sign = prop.delta >= 0 ? "+" : "";
      const deltaStr =
        typeof prop.delta === "number" && prop.delta % 1 !== 0
          ? `${sign}${prop.delta.toFixed(2)}`
          : `${sign}${prop.delta}`;
      ctx.fillStyle = prop.delta >= 0 ? "#4ade80" : "#f87171";
      ctx.font = "10px monospace";
      ctx.textAlign = "left";
      ctx.fillText(`Δ ${deltaStr}`, px + 4, deltaY);
    }
  }
}
