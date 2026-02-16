import { createReactBlockSpec } from "@blocknote/react";
import { useFullMetrics } from "../../../hooks/useMetrics.js";
import { useExperiments } from "../../../hooks/useExperiments.js";
import { defaultProps } from "@blocknote/core";

export const ExperimentChartBlock = createReactBlockSpec(
  {
    type: "experimentChart" as const,
    propSchema: {
      ...defaultProps,
      experimentId: { default: "" },
      metricKey: { default: "Losses/Total" },
    },
    content: "none",
  },
  {
    render: ({ block, editor }: any) => {
      const { experimentId, metricKey } = block.props;
      const { data: expData } = useExperiments();
      const { data: metricsData, isLoading } = useFullMetrics(
        experimentId || null,
      );

      const experiments = expData?.experiments ?? [];
      const points = metricsData?.points ?? [];
      const isEditable = editor.isEditable;

      return (
        <div className="my-2 rounded-lg border border-gray-700 bg-gray-900 p-4">
          {isEditable && (
            <div className="mb-3 flex flex-wrap items-center gap-2">
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
                value={metricKey}
                onChange={(e) => {
                  editor.updateBlock(block, {
                    props: { metricKey: e.target.value },
                  });
                }}
              >
                <option value="Losses/Total">Loss</option>
                <option value="Learning Rate">Learning Rate</option>
                <option value="Perplexity">Perplexity</option>
                <option value="Iterations/Second">Throughput</option>
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
