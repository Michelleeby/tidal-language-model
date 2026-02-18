import { useEffect, useRef } from "react";
import { createReactBlockSpec } from "@blocknote/react";
import { defaultProps } from "@blocknote/core";
import { useExperiments } from "../../../hooks/useExperiments.js";
import { useCheckpoints } from "../../../hooks/useMetrics.js";
import { useTrajectoryAnalysis } from "../../../hooks/useTrajectoryAnalysis.js";
import type { SweepAnalysis, AnalyzeRequest } from "@tidal/shared";

const SIGNAL_NAMES = ["modulation"] as const;
const SIGNAL_COLORS = ["#a78bfa"] as const;
const TEXT_PROPS = ["wordCount", "uniqueTokenRatio", "charCount"] as const;
const PROP_LABELS: Record<string, string> = {
  wordCount: "Words",
  uniqueTokenRatio: "Unique Ratio",
  charCount: "Chars",
};

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

// ---------------------------------------------------------------------------
// Canvas rendering
// ---------------------------------------------------------------------------

const PANEL_W = 600;
const ROW_H = 80;
const HEADER_H = 24;

function renderSweepPanel(
  canvas: HTMLCanvasElement | null,
  data: SweepSignalData[],
) {
  if (!canvas || data.length === 0) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const totalH = HEADER_H + data.length * ROW_H;
  const dpr = window.devicePixelRatio || 1;
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
      PROP_LABELS[TEXT_PROPS[j]] ?? TEXT_PROPS[j],
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
    },
    content: "none",
  },
  {
    render: ({ block, editor }: any) => {
      const { experimentId, prompt } = block.props;
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
        analysis.mutate(body);
      };

      const sweep = analysis.data?.sweepAnalysis;

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
              Running 3-config extreme-value sweep...
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
