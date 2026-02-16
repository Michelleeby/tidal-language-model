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
              key={ablationMetricKey}
              results={ablationData?.results ?? null}
              metricKey={ablationMetricKey}
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
  "episode_lengths",
  "policy_loss",
  "value_loss",
  "entropy",
]);

/** Extract a numeric array from an RLTrainingHistory by key. */
export function extractRLValues(
  history: RLTrainingHistory | null,
  key: string,
): number[] {
  if (!history || !RL_KEYS.has(key)) return [];
  return (history[key as keyof RLTrainingHistory] as number[]).filter(
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

/** Return the metric select options for a given chart mode. */
export function metricOptionsForMode(
  mode: string,
): Array<{ value: string; label: string }> {
  switch (mode) {
    case "rl":
      return [
        { value: "episode_rewards", label: "Episode Rewards" },
        { value: "episode_lengths", label: "Episode Lengths" },
        { value: "policy_loss", label: "Policy Loss" },
        { value: "value_loss", label: "Value Loss" },
        { value: "entropy", label: "Entropy" },
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
  episode_lengths: "#14b8a6",
  policy_loss: "#ef4444",
  value_loss: "#3b82f6",
  entropy: "#f59e0b",
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

/** Canvas grouped bar chart for ablation study results. */
function MiniAblationChart({
  results,
  metricKey,
}: {
  results: AblationResults | null;
  metricKey: string;
}) {
  const { labels, values, errors } = extractAblationValues(results, metricKey);

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

    ctx.clearRect(0, 0, BASE_W, BASE_H);

    const minVal = Math.min(...values);
    const maxVal = Math.max(...values.map((v, i) => v + (errors?.[i] ?? 0)));
    const range = maxVal - minVal || 1;
    const barPadding = 12;
    const labelArea = 20;
    const chartH = BASE_H - labelArea - 8;
    const barWidth = Math.min(60, (BASE_W - barPadding * 2) / values.length - barPadding);
    const totalBarsWidth = values.length * (barWidth + barPadding) - barPadding;
    const startX = (BASE_W - totalBarsWidth) / 2;

    // Grid lines
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 4; i++) {
      const y = 8 + (chartH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(BASE_W, y);
      ctx.stroke();
    }

    for (let i = 0; i < values.length; i++) {
      const x = startX + i * (barWidth + barPadding);
      const barH = ((values[i] - minVal) / range) * chartH;
      const y = 8 + chartH - barH;

      // Bar
      ctx.fillStyle = "#3b82f6";
      ctx.fillRect(x, y, barWidth, barH);

      // Error bar
      if (errors?.[i] != null) {
        const errH = (errors[i] / range) * chartH;
        const cx = x + barWidth / 2;
        ctx.strokeStyle = "#9ca3af";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx, y - errH);
        ctx.lineTo(cx, y + errH);
        ctx.stroke();
        // Caps
        ctx.beginPath();
        ctx.moveTo(cx - 3, y - errH);
        ctx.lineTo(cx + 3, y - errH);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx - 3, y + errH);
        ctx.lineTo(cx + 3, y + errH);
        ctx.stroke();
      }

      // Label
      ctx.fillStyle = "#9ca3af";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(
        labels[i].length > 10 ? labels[i].slice(0, 10) + "..." : labels[i],
        x + barWidth / 2,
        BASE_H - 4,
      );
    }
    ctx.textAlign = "start";

    // Y-axis labels
    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px monospace";
    ctx.fillText(maxVal.toFixed(2), 4, 16);
    ctx.fillText(minVal.toFixed(2), 4, 8 + chartH);
  };

  if (values.length === 0) {
    return (
      <div className="text-gray-500 text-sm py-8 text-center">
        No ablation data available for {metricKey}
      </div>
    );
  }

  const label = metricOptionsForMode("ablation").find((o) => o.value === metricKey)?.label ?? metricKey;

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
        {labels.length} policies
      </div>
    </div>
  );
}
