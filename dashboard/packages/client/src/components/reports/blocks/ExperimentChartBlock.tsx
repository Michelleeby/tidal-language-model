import { createReactBlockSpec } from "@blocknote/react";
import { useFullMetrics, useRLMetrics, useAblation } from "../../../hooks/useMetrics.js";
import { useExperiments } from "../../../hooks/useExperiments.js";
import { defaultProps } from "@blocknote/core";
import type { RLTrainingHistory, AblationResults } from "@tidal/shared";

export const ExperimentChartBlock = createReactBlockSpec(
  {
    type: "experimentChart" as const,
    propSchema: {
      ...defaultProps,
      experimentId: { default: "" },
      metricKey: { default: "Losses/Total" },
      chartMode: { default: "lm" },
      rlMetricKey: { default: "episode_rewards" },
      ablationMetricKey: { default: "mean_reward" },
    },
    content: "none",
  },
  {
    render: ({ block, editor }: any) => {
      const { experimentId, metricKey, chartMode, rlMetricKey, ablationMetricKey } = block.props;
      const { data: expData } = useExperiments();
      const { data: metricsData, isLoading: lmLoading } = useFullMetrics(
        experimentId || null,
      );
      const { data: rlData, isLoading: rlLoading } = useRLMetrics(
        experimentId || null,
      );
      const { data: ablationData, isLoading: ablationLoading } = useAblation(
        experimentId || null,
      );

      const experiments = expData?.experiments ?? [];
      const points = metricsData?.points ?? [];
      const isEditable = editor.isEditable;

      const activeMetricKey =
        chartMode === "rl" ? rlMetricKey :
        chartMode === "ablation" ? ablationMetricKey :
        metricKey;

      const isLoading =
        chartMode === "rl" ? rlLoading :
        chartMode === "ablation" ? ablationLoading :
        lmLoading;

      const modeOptions = metricOptionsForMode(chartMode);

      const metricPropName =
        chartMode === "rl" ? "rlMetricKey" :
        chartMode === "ablation" ? "ablationMetricKey" :
        "metricKey";

      return (
        <div className="my-2 rounded-lg border border-gray-700 bg-gray-900 p-4">
          {isEditable && (
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <select
                className="rounded bg-gray-800 border border-gray-600 text-gray-200 text-sm px-2 py-1"
                value={chartMode}
                onChange={(e) => {
                  editor.updateBlock(block, {
                    props: { chartMode: e.target.value },
                  });
                }}
              >
                <option value="lm">LM Training</option>
                <option value="rl">RL Training</option>
                <option value="ablation">Ablation</option>
              </select>
              <select
                className="rounded bg-gray-800 border border-gray-600 text-gray-200 text-sm px-2 py-1"
                value={experimentId}
                onChange={(e) => {
                  editor.updateBlock(block, {
                    props: { experimentId: e.target.value },
                  });
                }}
              >
                <option value="">Select experiment...</option>
                {experiments.map((exp) => (
                  <option key={exp.id} value={exp.id}>
                    {exp.id.length > 24 ? `${exp.id.slice(0, 24)}...` : exp.id}
                  </option>
                ))}
              </select>
              {chartMode !== "ablation" && (
                <select
                  className="rounded bg-gray-800 border border-gray-600 text-gray-200 text-sm px-2 py-1"
                  value={activeMetricKey}
                  onChange={(e) => {
                    editor.updateBlock(block, {
                      props: { [metricPropName]: e.target.value },
                    });
                  }}
                >
                  {modeOptions.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              )}
            </div>
          )}

          {!experimentId ? (
            <div className="text-gray-500 text-sm py-8 text-center">
              Select an experiment to display chart
            </div>
          ) : isLoading ? (
            <div className="text-gray-500 text-sm py-8 text-center">
              Loading metrics...
            </div>
          ) : chartMode === "rl" ? (
            <MiniRLChart
              key={rlMetricKey}
              history={rlData?.metrics?.history ?? null}
              metricKey={rlMetricKey}
            />
          ) : chartMode === "ablation" ? (
            <MiniAblationChart
              results={ablationData?.results ?? null}
            />
          ) : points.length === 0 ? (
            <div className="text-gray-500 text-sm py-8 text-center">
              No data available for {metricKey}
            </div>
          ) : (
            <MiniChart key={metricKey} points={points} metricKey={metricKey} />
          )}
        </div>
      );
    },
  },
);

/** Extract numeric values from metric points, handling derived metrics. */
export function extractValues(
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

const RL_KEYS: ReadonlySet<string> = new Set([
  "episode_rewards",
  "policy_loss",
  "value_loss",
  "entropy",
  "gate_creativity",
  "gate_focus",
  "gate_stability",
  "reward_perplexity",
  "reward_diversity",
  "reward_focus",
  "reward_repetition",
  "reward_coherence",
  "explained_variance",
]);

/** Extract a numeric array from an RLTrainingHistory by key. */
export function extractRLValues(
  history: RLTrainingHistory | null,
  key: string,
): number[] {
  if (!history || !RL_KEYS.has(key)) return [];
  return ((history[key as keyof RLTrainingHistory] as number[] | undefined) ?? []).filter(
    (v) => !isNaN(v),
  );
}

/** Extract labels, values, and optional error bars from AblationResults. */
export function extractAblationValues(
  results: AblationResults | null,
  metricKey: string,
): { labels: string[]; values: number[]; errors?: number[] } {
  if (!results) return { labels: [], values: [] };
  const labels = Object.keys(results);
  const values = labels.map((l) => (results[l] as any)[metricKey] as number);
  if (metricKey === "mean_reward") {
    const errors = labels.map((l) => results[l].std_reward);
    return { labels, values, errors };
  }
  return { labels, values };
}

const ABLATION_SERIES = [
  { key: "mean_reward", label: "reward", color: "#3b82f6" },
  { key: "mean_diversity", label: "diversity", color: "#22c55e" },
  { key: "mean_perplexity", label: "perplexity", color: "#a855f7" },
] as const;

/** Extract all ablation metrics grouped by policy for a grouped bar chart. */
export function extractAllAblationMetrics(
  results: AblationResults | null,
): {
  policies: string[];
  series: Array<{ key: string; label: string; color: string; values: number[] }>;
} {
  if (!results) return { policies: [], series: [] };
  const policies = Object.keys(results);
  const series = ABLATION_SERIES.map((m) => ({
    ...m,
    values: policies.map((p) => (results[p] as any)[m.key] as number),
  }));
  return { policies, series };
}

/** Return the metric select options for a given chart mode. */
export function metricOptionsForMode(
  mode: string,
): Array<{ value: string; label: string }> {
  switch (mode) {
    case "rl":
      return [
        { value: "episode_rewards", label: "Episode Rewards" },
        { value: "policy_loss", label: "Policy Loss" },
        { value: "value_loss", label: "Value Loss" },
        { value: "entropy", label: "Entropy" },
        { value: "gate_creativity", label: "Gate: Creativity" },
        { value: "gate_focus", label: "Gate: Focus" },
        { value: "gate_stability", label: "Gate: Stability" },
        { value: "reward_perplexity", label: "Reward: Perplexity" },
        { value: "reward_diversity", label: "Reward: Diversity" },
        { value: "reward_focus", label: "Reward: Focus" },
        { value: "reward_repetition", label: "Reward: Repetition" },
        { value: "reward_coherence", label: "Reward: Coherence" },
        { value: "explained_variance", label: "Explained Variance" },
      ];
    case "ablation":
      return [
        { value: "mean_reward", label: "Mean Reward" },
        { value: "mean_diversity", label: "Mean Diversity" },
        { value: "mean_perplexity", label: "Mean Perplexity" },
      ];
    default:
      return [
        { value: "Losses/Total", label: "Loss" },
        { value: "Learning Rate", label: "Learning Rate" },
        { value: "Perplexity", label: "Perplexity" },
        { value: "Iterations/Second", label: "Throughput" },
      ];
  }
}

const BASE_W = 600;
const BASE_H = 200;

/** Lightweight inline chart using canvas. */
function MiniChart({
  points,
  metricKey,
}: {
  points: Array<Record<string, unknown>>;
  metricKey: string;
}) {
  const canvasRef = (canvas: HTMLCanvasElement | null) => {
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = BASE_W * dpr;
    canvas.height = BASE_H * dpr;
    canvas.style.width = `${BASE_W}px`;
    canvas.style.height = `${BASE_H}px`;
    ctx.scale(dpr, dpr);

    const values = extractValues(points, metricKey);

    if (values.length === 0) return;

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    ctx.clearRect(0, 0, BASE_W, BASE_H);

    // Grid lines
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 4; i++) {
      const y = (BASE_H / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(BASE_W, y);
      ctx.stroke();
    }

    // Data line
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = (i / (values.length - 1)) * BASE_W;
      const y = BASE_H - ((values[i] - min) / range) * (BASE_H - 8) - 4;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Labels
    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px monospace";
    ctx.fillText(max.toFixed(4), 4, 12);
    ctx.fillText(min.toFixed(4), 4, BASE_H - 4);
  };

  return (
    <div>
      <div className="text-xs text-gray-400 mb-1 font-mono">{metricKey}</div>
      <canvas
        ref={canvasRef}
        width={BASE_W}
        height={BASE_H}
        className="w-full rounded bg-gray-950"
        style={{ height: BASE_H }}
      />
      <div className="text-xs text-gray-500 mt-1">
        {points.length} data points
      </div>
    </div>
  );
}

const RL_COLORS: Record<string, string> = {
  episode_rewards: "#22c55e",
  policy_loss: "#ef4444",
  value_loss: "#3b82f6",
  entropy: "#f59e0b",
  gate_creativity: "#f472b6",
  gate_focus: "#60a5fa",
  gate_stability: "#34d399",
  reward_perplexity: "#c084fc",
  reward_diversity: "#4ade80",
  reward_repetition: "#fb923c",
  reward_coherence: "#38bdf8",
  explained_variance: "#a78bfa",
};

/** Canvas line chart for RL training metrics. */
function MiniRLChart({
  history,
  metricKey,
}: {
  history: RLTrainingHistory | null;
  metricKey: string;
}) {
  const values = extractRLValues(history, metricKey);

  const canvasRef = (canvas: HTMLCanvasElement | null) => {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = BASE_W * dpr;
    canvas.height = BASE_H * dpr;
    canvas.style.width = `${BASE_W}px`;
    canvas.style.height = `${BASE_H}px`;
    ctx.scale(dpr, dpr);

    if (values.length === 0) return;

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    ctx.clearRect(0, 0, BASE_W, BASE_H);

    // Grid lines
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 4; i++) {
      const y = (BASE_H / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(BASE_W, y);
      ctx.stroke();
    }

    // Data line
    ctx.strokeStyle = RL_COLORS[metricKey] ?? "#3b82f6";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = (i / (values.length - 1)) * BASE_W;
      const y = BASE_H - ((values[i] - min) / range) * (BASE_H - 8) - 4;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Labels
    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px monospace";
    ctx.fillText(max.toFixed(4), 4, 12);
    ctx.fillText(min.toFixed(4), 4, BASE_H - 4);
  };

  if (values.length === 0) {
    return (
      <div className="text-gray-500 text-sm py-8 text-center">
        No RL data available for {metricKey}
      </div>
    );
  }

  const label = metricOptionsForMode("rl").find((o) => o.value === metricKey)?.label ?? metricKey;

  return (
    <div>
      <div className="text-xs text-gray-400 mb-1 font-mono">{label}</div>
      <canvas
        ref={canvasRef}
        width={BASE_W}
        height={BASE_H}
        className="w-full rounded bg-gray-950"
        style={{ height: BASE_H }}
      />
      <div className="text-xs text-gray-500 mt-1">
        {values.length} episodes
      </div>
    </div>
  );
}

const ABLATION_H = 250;

/** Canvas grouped bar chart for ablation study results — shows all metrics. */
function MiniAblationChart({
  results,
}: {
  results: AblationResults | null;
}) {
  const { policies, series } = extractAllAblationMetrics(results);

  // Collect every value across all series to find the global range.
  const allValues = series.flatMap((s) => s.values);

  const canvasRef = (canvas: HTMLCanvasElement | null) => {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = BASE_W * dpr;
    canvas.height = ABLATION_H * dpr;
    canvas.style.width = `${BASE_W}px`;
    canvas.style.height = `${ABLATION_H}px`;
    ctx.scale(dpr, dpr);

    if (allValues.length === 0) return;

    ctx.clearRect(0, 0, BASE_W, ABLATION_H);

    const dataMin = Math.min(...allValues);
    const dataMax = Math.max(...allValues);
    // Ensure the zero line is always visible in the chart range.
    const rangeMin = Math.min(dataMin, 0);
    const rangeMax = Math.max(dataMax, 0);
    const range = rangeMax - rangeMin || 1;

    const yAxisW = 44;
    const labelArea = 22;
    const legendH = 20;
    const chartTop = 8;
    const chartH = ABLATION_H - labelArea - chartTop - legendH;
    const chartW = BASE_W - yAxisW;

    /** Map a data value to a y pixel. */
    const toY = (v: number) => chartTop + chartH - ((v - rangeMin) / range) * chartH;
    const zeroY = toY(0);

    // Grid lines at nice intervals
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 0.5;
    const gridSteps = 4;
    for (let i = 0; i <= gridSteps; i++) {
      const v = rangeMin + (range / gridSteps) * i;
      const y = toY(v);
      ctx.beginPath();
      ctx.moveTo(yAxisW, y);
      ctx.lineTo(BASE_W, y);
      ctx.stroke();

      // Y-axis tick labels
      ctx.fillStyle = "#9ca3af";
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      ctx.fillText(v.toFixed(1), yAxisW - 4, y + 3);
    }

    // Zero line (heavier)
    ctx.strokeStyle = "#6b7280";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(yAxisW, zeroY);
    ctx.lineTo(BASE_W, zeroY);
    ctx.stroke();

    // Grouped bars
    const numSeries = series.length;
    const groupPadding = 24;
    const groupW = (chartW - groupPadding) / policies.length;
    const barGap = 2;
    const barWidth = Math.min(
      24,
      (groupW - groupPadding - barGap * (numSeries - 1)) / numSeries,
    );
    const barsBlockW = numSeries * barWidth + (numSeries - 1) * barGap;

    for (let pi = 0; pi < policies.length; pi++) {
      const groupX = yAxisW + groupPadding / 2 + pi * groupW;
      const barsStartX = groupX + (groupW - barsBlockW) / 2;

      for (let si = 0; si < numSeries; si++) {
        const val = series[si].values[pi];
        const x = barsStartX + si * (barWidth + barGap);
        const barTop = val >= 0 ? toY(val) : zeroY;
        const barH = Math.abs(val) / range * chartH;

        ctx.fillStyle = series[si].color;
        ctx.fillRect(x, barTop, barWidth, barH);
      }

      // Policy label
      ctx.fillStyle = "#9ca3af";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      const labelX = groupX + groupW / 2;
      const labelY = chartTop + chartH + 14;
      const name = policies[pi];
      ctx.fillText(
        name.length > 12 ? name.slice(0, 12) + "..." : name,
        labelX,
        labelY,
      );
    }

    // Legend
    const legendY = ABLATION_H - 6;
    let lx = yAxisW + 8;
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    for (const s of series) {
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, legendY - 8, 10, 10);
      ctx.fillStyle = "#9ca3af";
      ctx.fillText(s.label, lx + 14, legendY);
      lx += ctx.measureText(s.label).width + 30;
    }
  };

  if (policies.length === 0) {
    return (
      <div className="text-gray-500 text-sm py-8 text-center">
        No ablation data available
      </div>
    );
  }

  return (
    <div>
      <div className="text-xs text-gray-400 mb-1 font-mono">Ablation Comparison</div>
      <canvas
        ref={canvasRef}
        width={BASE_W}
        height={ABLATION_H}
        className="w-full rounded bg-gray-950"
        style={{ height: ABLATION_H }}
      />
      <div className="text-xs text-gray-500 mt-1">
        {policies.length} policies
      </div>
    </div>
  );
}
