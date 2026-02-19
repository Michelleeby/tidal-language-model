import { useEffect, useRef } from "react";
import { createReactBlockSpec } from "@blocknote/react";
import { defaultProps } from "@blocknote/core";
import { useExperiments } from "../../../hooks/useExperiments.js";
import { useCheckpoints } from "../../../hooks/useMetrics.js";
import { useTrajectoryAnalysis } from "../../../hooks/useTrajectoryAnalysis.js";
import { useAnalyses, useAnalysis, useSaveAnalysis } from "../../../hooks/useAnalyses.js";
import { gatingModeOptions } from "./TrajectoryChartBlock.js";
import {
  renderHeatmap,
  SIGNAL_NAMES,
  SIGNAL_COLORS,
  HEATMAP_DIMENSIONS,
} from "../../../utils/chartRenderers.js";
import type { BatchAnalysis, AnalyzeRequest, AnalyzeResponse } from "@tidal/shared";
import { CURATED_PROMPTS } from "@tidal/shared";

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

const HEATMAP_W = HEATMAP_DIMENSIONS.width;
const HEATMAP_H = HEATMAP_DIMENSIONS.height;

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
      analysisId: { default: "" },
    },
    content: "none",
  },
  {
    render: ({ block, editor }: any) => {
      const { experimentId, gatingMode, analysisId } = block.props;
      const { data: expData } = useExperiments();
      const { data: checkpointsData } = useCheckpoints(experimentId || null);
      const { data: cachedAnalyses } = useAnalyses(experimentId || null, "cross-prompt");
      const { data: cachedData } = useAnalysis(analysisId || null);
      const saveAnalysis = useSaveAnalysis();
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
        analysis.mutate(body, {
          onSuccess: (data) => {
            if (experimentId) {
              saveAnalysis.mutate(
                {
                  expId: experimentId,
                  analysisType: "cross-prompt",
                  label: `Cross-prompt — ${gatingMode} — ${CURATED_PROMPTS.length} prompts`,
                  request: body as unknown as Record<string, unknown>,
                  data: data as unknown as Record<string, unknown>,
                },
                {
                  onSuccess: (resp) => {
                    editor.updateBlock(block, {
                      props: { analysisId: resp.analysis.id },
                    });
                  },
                },
              );
            }
          },
        });
      };

      // Use cached data when available
      const cachedBatch = cachedData?.analysis?.data
        ? (cachedData.analysis.data as unknown as AnalyzeResponse).batchAnalysis
        : undefined;

      // Reset live mutation when user selects a different cached analysis,
      // otherwise analysis.data permanently shadows the cached selection.
      useEffect(() => {
        if (analysisId) {
          analysis.reset();
        }
      // eslint-disable-next-line react-hooks/exhaustive-deps
      }, [analysisId]);

      const batch = analysis.data?.batchAnalysis ?? cachedBatch;

      // Re-render canvas whenever analysis data changes
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
              {(cachedAnalyses?.analyses ?? []).length > 0 && (
                <select
                  className="rounded bg-gray-800 border border-gray-600 text-gray-200 text-xs px-2 py-1"
                  value={analysisId}
                  onChange={(e) => {
                    editor.updateBlock(block, {
                      props: { analysisId: e.target.value },
                    });
                  }}
                >
                  <option value="">Cached analyses...</option>
                  {(cachedAnalyses?.analyses ?? []).map((a) => (
                    <option key={a.id} value={a.id}>
                      {a.label}
                    </option>
                  ))}
                </select>
              )}
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
