/**
 * Shared chart data extraction functions used by both report blocks and export.
 * Pure functions only — no DOM or Canvas dependencies — so they're testable
 * and importable in both browser and test environments.
 *
 * Canvas rendering functions accept a CanvasRenderingContext2D and draw directly,
 * used by both block components and HTML export serialization.
 */

import type {
  BatchAnalysis,
  SweepAnalysis,
  RLTrainingHistory,
  AblationResults,
} from "@tidal/shared";

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
// Heatmap (cross-prompt analysis)
// ---------------------------------------------------------------------------

const SIGNAL_NAMES = ["modulation"] as const;

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

// ---------------------------------------------------------------------------
// Sweep panel
// ---------------------------------------------------------------------------

const TEXT_PROPS = ["wordCount", "uniqueTokenRatio", "charCount"] as const;

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
