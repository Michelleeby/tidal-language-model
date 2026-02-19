import { useEffect, useRef } from "react";
import { createReactBlockSpec } from "@blocknote/react";
import { defaultProps } from "@blocknote/core";
import { useExperiments } from "../../../hooks/useExperiments.js";
import { useCheckpoints } from "../../../hooks/useMetrics.js";
import { useTrajectoryAnalysis } from "../../../hooks/useTrajectoryAnalysis.js";
import { useAnalyses, useAnalysis, useSaveAnalysis } from "../../../hooks/useAnalyses.js";
import {
  renderSweepPanel,
  SIGNAL_NAMES,
  SIGNAL_COLORS,
  SWEEP_DIMENSIONS,
  TEXT_PROP_LABELS,
} from "../../../utils/chartRenderers.js";
import type { SweepAnalysis, AnalyzeRequest, AnalyzeResponse } from "@tidal/shared";

// ---------------------------------------------------------------------------
// Pure data extraction functions (exported for testing)
// ---------------------------------------------------------------------------

export interface SweepPropertyData {
  name: string;
  low: number;
  high: number;
  delta: number;
}

export interface SweepSignalData {
  signal: string;
  properties: SweepPropertyData[];
}

/** Extract per-signal low/high/delta for each text property. */
export function extractSweepPanelData(sweep: SweepAnalysis): SweepSignalData[] {
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
        const label = PROP_LABELS[prop] ?? prop;
        return `${label}: ${sign}${typeof d === "number" && d % 1 !== 0 ? d.toFixed(2) : d}`;
      });
    return `${signal} low→high: ${parts.join(", ")}`;
  });
}

const TEXT_PROPS = ["wordCount", "uniqueTokenRatio", "charCount"] as const;
const PROP_LABELS = TEXT_PROP_LABELS;
const PANEL_W = SWEEP_DIMENSIONS.width;
const ROW_H = SWEEP_DIMENSIONS.rowHeight;
const HEADER_H = SWEEP_DIMENSIONS.headerHeight;

// ---------------------------------------------------------------------------
// Block component
// ---------------------------------------------------------------------------

export const SweepBlock = createReactBlockSpec(
  {
    type: "sweepAnalysis" as const,
    propSchema: {
      ...defaultProps,
      experimentId: { default: "" },
      prompt: { default: "Once upon a time," },
      analysisId: { default: "" },
    },
    content: "none",
  },
  {
    render: ({ block, editor }: any) => {
      const { experimentId, prompt, analysisId } = block.props;
      const { data: expData } = useExperiments();
      const { data: checkpointsData } = useCheckpoints(experimentId || null);
      const { data: cachedAnalyses } = useAnalyses(experimentId || null, "sweep");
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

      const canvasRef = useRef<HTMLCanvasElement>(null);

      const handleSweep = () => {
        if (!checkpoint) return;
        const body: AnalyzeRequest = {
          checkpoint,
          prompts: [prompt],
          maxTokens: 50,
          gatingMode: "fixed",
          includeExtremeValues: true,
        };
        analysis.mutate(body, {
          onSuccess: (data) => {
            if (experimentId) {
              saveAnalysis.mutate(
                {
                  expId: experimentId,
                  analysisType: "sweep",
                  label: `Sweep — "${prompt.slice(0, 30)}"`,
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
      const cachedSweep = cachedData?.analysis?.data
        ? (cachedData.analysis.data as unknown as AnalyzeResponse).sweepAnalysis
        : undefined;

      // Reset live mutation when user selects a different cached analysis,
      // otherwise analysis.data permanently shadows the cached selection.
      useEffect(() => {
        if (analysisId) {
          analysis.reset();
        }
      // eslint-disable-next-line react-hooks/exhaustive-deps
      }, [analysisId]);

      const sweep = analysis.data?.sweepAnalysis ?? cachedSweep;

      // Re-render canvas whenever sweep data changes or canvas mounts
      useEffect(() => {
        if (!sweep) return;
        const panelData = extractSweepPanelData(sweep);
        renderSweepPanel(canvasRef.current, panelData);
      }, [sweep]);
      const summaryLines = sweep
        ? extractInterpretabilitySummary(sweep.interpretabilityMap)
        : [];
      const panelData = sweep ? extractSweepPanelData(sweep) : [];
      const panelH = panelData.length > 0 ? HEADER_H + panelData.length * ROW_H : 200;

      return (
        <div className="my-2 rounded-lg border border-gray-700 bg-gray-900 p-4">
          {isEditable && (
            <div className="mb-3 space-y-2">
              <div className="flex flex-wrap items-center gap-2">
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
                <button
                  onClick={handleSweep}
                  disabled={!checkpoint || analysis.isPending}
                  className="px-3 py-1 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {analysis.isPending ? "Sweeping..." : "Run Sweep"}
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
              <textarea
                className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200 resize-none"
                rows={2}
                value={prompt}
                onChange={(e) => {
                  editor.updateBlock(block, {
                    props: { prompt: e.target.value },
                  });
                }}
                placeholder="Enter prompt for sweep..."
              />
            </div>
          )}

          {!experimentId ? (
            <div className="text-gray-500 text-sm py-8 text-center">
              Select an experiment to run a gate signal sweep
            </div>
          ) : analysis.isPending ? (
            <div className="text-gray-500 text-sm py-8 text-center">
              Running 15-config extreme-value sweep...
            </div>
          ) : sweep ? (
            <div className="space-y-4">
              <div>
                <div className="text-xs text-gray-400 mb-1 font-mono">
                  Gate Signal Sweep — Low vs High
                </div>
                <canvas
                  ref={canvasRef}
                  width={PANEL_W}
                  height={panelH}
                  className="w-full rounded bg-gray-950"
                  style={{ height: panelH }}
                />
              </div>
              {summaryLines.length > 0 && (
                <div className="text-xs text-gray-400 font-mono space-y-1">
                  <div className="font-medium text-gray-300">Interpretability Summary</div>
                  {summaryLines.map((line, i) => (
                    <div key={i}>
                      <span style={{ color: SIGNAL_COLORS[i] }}>
                        {line.split(" low")[0]}
                      </span>
                      {" low" + line.split(" low").slice(1).join(" low")}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="text-gray-500 text-sm py-8 text-center">
              Click &ldquo;Run Sweep&rdquo; to test extreme gate values
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
