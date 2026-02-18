import { useEffect, useRef } from "react";
import { createReactBlockSpec } from "@blocknote/react";
import { defaultProps } from "@blocknote/core";
import { useExperiments } from "../../../hooks/useExperiments.js";
import { useCheckpoints } from "../../../hooks/useMetrics.js";
import { useTrajectoryAnalysis } from "../../../hooks/useTrajectoryAnalysis.js";
import { gatingModeOptions } from "./TrajectoryChartBlock.js";
import type { BatchAnalysis, AnalyzeRequest } from "@tidal/shared";
import { CURATED_PROMPTS } from "@tidal/shared";

const SIGNAL_NAMES = ["modulation"] as const;
const SIGNAL_COLORS = ["#a78bfa"] as const;

// ---------------------------------------------------------------------------
// Pure data extraction functions (exported for testing)
// ---------------------------------------------------------------------------

/** Extract heatmap data: rows = prompts, columns = signals, values = means. */
export function extractHeatmapData(batch: BatchAnalysis): {
  prompts: string[];
  signals: string[];
  values: number[][];
} {
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
// Canvas rendering helpers
// ---------------------------------------------------------------------------

const HEATMAP_W = 600;
const HEATMAP_H = 300;

function renderHeatmap(
  canvas: HTMLCanvasElement | null,
  data: ReturnType<typeof extractHeatmapData>,
) {
  if (!canvas || data.prompts.length === 0) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  canvas.width = HEATMAP_W * dpr;
  canvas.height = HEATMAP_H * dpr;
  canvas.style.width = `${HEATMAP_W}px`;
  canvas.style.height = `${HEATMAP_H}px`;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, HEATMAP_W, HEATMAP_H);

  const labelW = 180;
  const headerH = 24;
  const cellW = (HEATMAP_W - labelW) / data.signals.length;
  const cellH = Math.min(24, (HEATMAP_H - headerH) / data.prompts.length);

  // Column headers
  ctx.font = "11px monospace";
  ctx.textAlign = "center";
  for (let j = 0; j < data.signals.length; j++) {
    ctx.fillStyle = SIGNAL_COLORS[j];
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

      // Color: blend from dark to signal color based on value
      const r = parseInt(SIGNAL_COLORS[j].slice(1, 3), 16);
      const g = parseInt(SIGNAL_COLORS[j].slice(3, 5), 16);
      const b = parseInt(SIGNAL_COLORS[j].slice(5, 7), 16);
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
// Block component
// ---------------------------------------------------------------------------

export const CrossPromptBlock = createReactBlockSpec(
  {
    type: "crossPromptAnalysis" as const,
    propSchema: {
      ...defaultProps,
      experimentId: { default: "" },
      gatingMode: { default: "fixed" },
    },
    content: "none",
  },
  {
    render: ({ block, editor }: any) => {
      const { experimentId, gatingMode } = block.props;
      const { data: expData } = useExperiments();
      const { data: checkpointsData } = useCheckpoints(experimentId || null);
      const isEditable = editor.isEditable;
      const analysis = useTrajectoryAnalysis();

      const experiments = expData?.experiments ?? [];
      const checkpoints = checkpointsData?.checkpoints ?? [];
      const checkpoint =
        checkpoints.find((cp) => cp.phase === "foundational")?.path ??
        checkpoints[0]?.path ??
        "";

      const heatmapCanvasRef = useRef<HTMLCanvasElement>(null);

      const handleAnalyze = () => {
        if (!checkpoint) return;
        const body: AnalyzeRequest = {
          checkpoint,
          prompts: CURATED_PROMPTS,
          maxTokens: 50,
          gatingMode: gatingMode as AnalyzeRequest["gatingMode"],
        };
        analysis.mutate(body);
      };

      // Re-render canvas whenever analysis data changes
      const batch = analysis.data?.batchAnalysis;
      useEffect(() => {
        if (!batch) return;
        const heatmap = extractHeatmapData(batch);
        renderHeatmap(heatmapCanvasRef.current, heatmap);
      }, [batch]);

      const modes = gatingModeOptions();
      const varianceSummary = batch
        ? extractVarianceSummary(batch.crossPromptVariance)
        : [];
      const strategy = batch?.strategyCharacterization;

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
                value={gatingMode}
                onChange={(e) => {
                  editor.updateBlock(block, {
                    props: { gatingMode: e.target.value },
                  });
                }}
              >
                {modes.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
              <button
                onClick={handleAnalyze}
                disabled={!checkpoint || analysis.isPending}
                className="px-3 py-1 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {analysis.isPending ? "Analyzing..." : "Run Analysis"}
              </button>
            </div>
          )}

          {!experimentId ? (
            <div className="text-gray-500 text-sm py-8 text-center">
              Select an experiment to run cross-prompt analysis
            </div>
          ) : analysis.isPending ? (
            <div className="text-gray-500 text-sm py-8 text-center">
              Analyzing {CURATED_PROMPTS.length} prompts...
            </div>
          ) : batch ? (
            <div className="space-y-4">
              <div>
                <div className="text-xs text-gray-400 mb-1 font-mono">
                  Signal Mean Heatmap ({Object.keys(batch.perPromptSummaries).length} prompts)
                </div>
                <canvas
                  ref={heatmapCanvasRef}
                  width={HEATMAP_W}
                  height={HEATMAP_H}
                  className="w-full rounded bg-gray-950"
                  style={{ height: HEATMAP_H }}
                />
              </div>
              {varianceSummary.length > 0 && (
                <div className="text-xs text-gray-400 font-mono space-y-1">
                  <div className="font-medium text-gray-300">Cross-Prompt Variance</div>
                  {varianceSummary.map((v) => (
                    <div key={v.signal}>
                      <span style={{ color: SIGNAL_COLORS[SIGNAL_NAMES.indexOf(v.signal as any)] }}>
                        {v.signal}
                      </span>
                      : between={v.between.toFixed(4)}, within={v.within.toFixed(4)}
                    </div>
                  ))}
                </div>
              )}
              {strategy && (
                <div className="text-xs text-gray-400 font-mono space-y-1">
                  <div className="font-medium text-gray-300">Strategy Characterization</div>
                  {SIGNAL_NAMES.map((s, i) => (
                    <div key={s}>
                      <span style={{ color: SIGNAL_COLORS[i] }}>{s}</span>
                      : mean={strategy[s]?.globalMean?.toFixed(4) ?? "N/A"},
                      std={strategy[s]?.globalStd?.toFixed(4) ?? "N/A"}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="text-gray-500 text-sm py-8 text-center">
              Click &ldquo;Run Analysis&rdquo; to analyze gating strategies across prompts
            </div>
          )}

          {analysis.isError && (
            <div className="bg-red-900/30 border border-red-800 rounded-lg p-3 text-sm text-red-300 mt-2">
              {(analysis.error as Error).message}
            </div>
          )}
        </div>
      );
    },
  },
);
