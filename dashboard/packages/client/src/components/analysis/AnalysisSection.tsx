import { useState, useEffect, useRef } from "react";
import { useCheckpoints } from "../../hooks/useMetrics.js";
import { useAnalyses, useSaveAnalysis, useDeleteAnalysis } from "../../hooks/useAnalyses.js";
import { useTrajectoryAnalysis } from "../../hooks/useTrajectoryAnalysis.js";
import { api } from "../../api/client.js";
import GateTrajectoryChart from "../charts/GateTrajectoryChart.js";
import { extractHeatmapData, extractVarianceSummary } from "../reports/blocks/CrossPromptBlock.js";
import { extractSweepPanelData, extractInterpretabilitySummary } from "../reports/blocks/SweepBlock.js";
import type {
  AnalysisType,
  AnalysisResultSummary,
  GenerateRequest,
  GenerateResponse,
  AnalyzeRequest,
  AnalyzeResponse,
  BatchAnalysis,
  SweepAnalysis,
} from "@tidal/shared";
import { CURATED_PROMPTS } from "@tidal/shared";
import { useMutation } from "@tanstack/react-query";

type TabType = "trajectory" | "cross-prompt" | "sweep";

interface AnalysisSectionProps {
  expId: string;
}

const SIGNAL_COLORS = ["#a78bfa"] as const;

export default function AnalysisSection({ expId }: AnalysisSectionProps) {
  const [activeTab, setActiveTab] = useState<TabType>("trajectory");

  const tabs: { key: TabType; label: string }[] = [
    { key: "trajectory", label: "Trajectory" },
    { key: "cross-prompt", label: "Cross-Prompt" },
    { key: "sweep", label: "Sweep" },
  ];

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex gap-1">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${
              activeTab === tab.key
                ? "bg-gray-700 text-gray-100"
                : "text-gray-500 hover:text-gray-300 hover:bg-gray-800"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "trajectory" && (
        <TrajectoryTab expId={expId} />
      )}
      {activeTab === "cross-prompt" && (
        <CrossPromptTab expId={expId} />
      )}
      {activeTab === "sweep" && (
        <SweepTab expId={expId} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared: cached analysis list
// ---------------------------------------------------------------------------

function CachedAnalysisList({
  expId,
  analysisType,
  onSelect,
}: {
  expId: string;
  analysisType: AnalysisType;
  onSelect?: (id: string) => void;
}) {
  const { data } = useAnalyses(expId, analysisType);
  const deleteAnalysis = useDeleteAnalysis();

  const analyses = data?.analyses ?? [];
  if (analyses.length === 0) return null;

  return (
    <div className="mt-3 border-t border-gray-800 pt-3">
      <div className="text-xs text-gray-400 font-medium mb-2">Cached Results</div>
      <div className="space-y-1">
        {analyses.map((a: AnalysisResultSummary) => (
          <div
            key={a.id}
            className="flex items-center justify-between bg-gray-800/50 rounded px-3 py-1.5 text-xs"
          >
            <button
              onClick={() => onSelect?.(a.id)}
              className="text-gray-300 hover:text-white truncate text-left flex-1"
            >
              {a.label}
              <span className="text-gray-500 ml-2">
                {new Date(a.createdAt).toLocaleString()}
              </span>
              <span className="text-gray-600 ml-2">
                ({(a.sizeBytes / 1024).toFixed(1)} KB)
              </span>
            </button>
            <button
              onClick={() => deleteAnalysis.mutate(a.id)}
              className="text-gray-500 hover:text-red-400 ml-2 flex-shrink-0"
              title="Delete"
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Trajectory tab
// ---------------------------------------------------------------------------

function TrajectoryTab({ expId }: { expId: string }) {
  const { data: checkpointsData } = useCheckpoints(expId);
  const saveAnalysis = useSaveAnalysis();
  const [gatingMode, setGatingMode] = useState<"fixed" | "random" | "learned">("fixed");
  const [prompt, setPrompt] = useState("Once upon a time,");
  const [result, setResult] = useState<GenerateResponse | null>(null);

  const checkpoints = checkpointsData?.checkpoints ?? [];
  const checkpoint =
    checkpoints.find((cp) => cp.phase === "foundational")?.path ??
    checkpoints[0]?.path ??
    "";

  const generateMutation = useMutation({
    mutationFn: (body: GenerateRequest) => api.generate(body),
    onSuccess: (data) => {
      setResult(data);
      // Auto-save to cache
      if (data.trajectory) {
        saveAnalysis.mutate({
          expId,
          analysisType: "trajectory",
          label: `Trajectory — ${gatingMode} — "${prompt.slice(0, 30)}"`,
          request: { checkpoint, prompt, gatingMode },
          data: data as unknown as Record<string, unknown>,
        });
      }
    },
  });

  const handleGenerate = () => {
    if (!checkpoint) return;
    generateMutation.mutate({
      checkpoint,
      prompt,
      maxTokens: 50,
      temperature: 0.8,
      topK: 50,
      gatingMode,
    });
  };

  const handleSelectCached = async (id: string) => {
    const resp = await api.getAnalysis(id);
    setResult(resp.analysis.data as unknown as GenerateResponse);
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <select
          className="rounded bg-gray-800 border border-gray-600 text-gray-200 text-sm px-2 py-1"
          value={gatingMode}
          onChange={(e) => setGatingMode(e.target.value as typeof gatingMode)}
        >
          <option value="fixed">Fixed</option>
          <option value="random">Random</option>
          <option value="learned">Learned (RL)</option>
        </select>
        <button
          onClick={handleGenerate}
          disabled={!checkpoint || generateMutation.isPending}
          className="px-3 py-1 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {generateMutation.isPending ? "Generating..." : "Generate Trajectory"}
        </button>
      </div>
      <textarea
        className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200 resize-none"
        rows={2}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Enter prompt..."
      />

      {generateMutation.isPending ? (
        <div className="text-gray-500 text-sm py-6 text-center">Generating trajectory...</div>
      ) : result?.trajectory ? (
        <GateTrajectoryChart trajectory={result.trajectory} />
      ) : (
        <div className="text-gray-500 text-sm py-6 text-center">
          Click &ldquo;Generate Trajectory&rdquo; to analyze gate behavior
        </div>
      )}

      {generateMutation.isError && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg p-3 text-sm text-red-300">
          {(generateMutation.error as Error).message}
        </div>
      )}

      <CachedAnalysisList
        expId={expId}
        analysisType="trajectory"
        onSelect={handleSelectCached}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Cross-Prompt tab
// ---------------------------------------------------------------------------

const HEATMAP_W = 600;
const HEATMAP_H = 300;

function CrossPromptTab({ expId }: { expId: string }) {
  const { data: checkpointsData } = useCheckpoints(expId);
  const saveAnalysis = useSaveAnalysis();
  const analysis = useTrajectoryAnalysis();
  const [gatingMode, setGatingMode] = useState<"fixed" | "random" | "learned">("fixed");
  const heatmapCanvasRef = useRef<HTMLCanvasElement>(null);

  const checkpoints = checkpointsData?.checkpoints ?? [];
  const checkpoint =
    checkpoints.find((cp) => cp.phase === "foundational")?.path ??
    checkpoints[0]?.path ??
    "";

  const handleAnalyze = () => {
    if (!checkpoint) return;
    const body: AnalyzeRequest = {
      checkpoint,
      prompts: CURATED_PROMPTS,
      maxTokens: 50,
      gatingMode,
    };
    analysis.mutate(body, {
      onSuccess: (data) => {
        saveAnalysis.mutate({
          expId,
          analysisType: "cross-prompt",
          label: `Cross-prompt — ${gatingMode} — ${CURATED_PROMPTS.length} prompts`,
          request: body as unknown as Record<string, unknown>,
          data: data as unknown as Record<string, unknown>,
        });
      },
    });
  };

  const batch = analysis.data?.batchAnalysis;

  useEffect(() => {
    if (!batch || !heatmapCanvasRef.current) return;
    const data = extractHeatmapData(batch);
    renderHeatmapCanvas(heatmapCanvasRef.current, data);
  }, [batch]);

  const varianceSummary = batch ? extractVarianceSummary(batch.crossPromptVariance) : [];

  const handleSelectCached = async (id: string) => {
    const resp = await api.getAnalysis(id);
    const cachedData = resp.analysis.data as unknown as AnalyzeResponse;
    // We need to update the analysis mutation data — simplest approach is to
    // re-render the heatmap directly
    if (cachedData.batchAnalysis && heatmapCanvasRef.current) {
      const heatmap = extractHeatmapData(cachedData.batchAnalysis);
      renderHeatmapCanvas(heatmapCanvasRef.current, heatmap);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <select
          className="rounded bg-gray-800 border border-gray-600 text-gray-200 text-sm px-2 py-1"
          value={gatingMode}
          onChange={(e) => setGatingMode(e.target.value as typeof gatingMode)}
        >
          <option value="fixed">Fixed</option>
          <option value="random">Random</option>
          <option value="learned">Learned (RL)</option>
        </select>
        <button
          onClick={handleAnalyze}
          disabled={!checkpoint || analysis.isPending}
          className="px-3 py-1 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {analysis.isPending ? "Analyzing..." : "Run Cross-Prompt Analysis"}
        </button>
      </div>

      {analysis.isPending ? (
        <div className="text-gray-500 text-sm py-6 text-center">
          Analyzing {CURATED_PROMPTS.length} prompts...
        </div>
      ) : batch ? (
        <div className="space-y-3">
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
                  <span style={{ color: SIGNAL_COLORS[0] }}>{v.signal}</span>
                  : between={v.between.toFixed(4)}, within={v.within.toFixed(4)}
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="text-gray-500 text-sm py-6 text-center">
          Click &ldquo;Run Cross-Prompt Analysis&rdquo; to analyze gating across prompts
        </div>
      )}

      {analysis.isError && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg p-3 text-sm text-red-300">
          {(analysis.error as Error).message}
        </div>
      )}

      <CachedAnalysisList
        expId={expId}
        analysisType="cross-prompt"
        onSelect={handleSelectCached}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sweep tab
// ---------------------------------------------------------------------------

const PANEL_W = 600;
const ROW_H = 80;
const HEADER_H = 24;
const TEXT_PROPS = ["wordCount", "uniqueTokenRatio", "charCount"] as const;
const PROP_LABELS: Record<string, string> = {
  wordCount: "Words",
  uniqueTokenRatio: "Unique Ratio",
  charCount: "Chars",
};

function SweepTab({ expId }: { expId: string }) {
  const { data: checkpointsData } = useCheckpoints(expId);
  const saveAnalysis = useSaveAnalysis();
  const analysis = useTrajectoryAnalysis();
  const [prompt, setPrompt] = useState("Once upon a time,");
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const checkpoints = checkpointsData?.checkpoints ?? [];
  const checkpoint =
    checkpoints.find((cp) => cp.phase === "foundational")?.path ??
    checkpoints[0]?.path ??
    "";

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
        saveAnalysis.mutate({
          expId,
          analysisType: "sweep",
          label: `Sweep — "${prompt.slice(0, 30)}"`,
          request: body as unknown as Record<string, unknown>,
          data: data as unknown as Record<string, unknown>,
        });
      },
    });
  };

  const sweep = analysis.data?.sweepAnalysis;

  useEffect(() => {
    if (!sweep || !canvasRef.current) return;
    const panelData = extractSweepPanelData(sweep);
    renderSweepCanvas(canvasRef.current, panelData);
  }, [sweep]);

  const summaryLines = sweep
    ? extractInterpretabilitySummary(sweep.interpretabilityMap)
    : [];
  const panelData = sweep ? extractSweepPanelData(sweep) : [];
  const panelH = panelData.length > 0 ? HEADER_H + panelData.length * ROW_H : 200;

  const handleSelectCached = async (id: string) => {
    const resp = await api.getAnalysis(id);
    const cachedData = resp.analysis.data as unknown as AnalyzeResponse;
    if (cachedData.sweepAnalysis && canvasRef.current) {
      const pd = extractSweepPanelData(cachedData.sweepAnalysis);
      renderSweepCanvas(canvasRef.current, pd);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={handleSweep}
          disabled={!checkpoint || analysis.isPending}
          className="px-3 py-1 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {analysis.isPending ? "Sweeping..." : "Run Extreme-Value Sweep"}
        </button>
      </div>
      <textarea
        className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200 resize-none"
        rows={2}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Enter prompt for sweep..."
      />

      {analysis.isPending ? (
        <div className="text-gray-500 text-sm py-6 text-center">
          Running extreme-value sweep...
        </div>
      ) : sweep ? (
        <div className="space-y-3">
          <div>
            <div className="text-xs text-gray-400 mb-1 font-mono">Gate Signal Sweep — Low vs High</div>
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
                  <span style={{ color: SIGNAL_COLORS[0] }}>{line.split(" low")[0]}</span>
                  {" low" + line.split(" low").slice(1).join(" low")}
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="text-gray-500 text-sm py-6 text-center">
          Click &ldquo;Run Extreme-Value Sweep&rdquo; to test extreme gate values
        </div>
      )}

      {analysis.isError && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg p-3 text-sm text-red-300">
          {(analysis.error as Error).message}
        </div>
      )}

      <CachedAnalysisList
        expId={expId}
        analysisType="sweep"
        onSelect={handleSelectCached}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Canvas rendering helpers (duplicated from block components to avoid
// coupling analysis section to BlockNote block specs)
// ---------------------------------------------------------------------------

function renderHeatmapCanvas(
  canvas: HTMLCanvasElement,
  data: { prompts: string[]; signals: string[]; values: number[][] },
) {
  if (data.prompts.length === 0) return;
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

  ctx.font = "11px monospace";
  ctx.textAlign = "center";
  for (let j = 0; j < data.signals.length; j++) {
    ctx.fillStyle = SIGNAL_COLORS[0];
    ctx.fillText(data.signals[j], labelW + j * cellW + cellW / 2, headerH - 6);
  }

  for (let i = 0; i < data.prompts.length; i++) {
    const y = headerH + i * cellH;
    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    const label = data.prompts[i].length > 28 ? data.prompts[i].slice(0, 28) + "..." : data.prompts[i];
    ctx.fillText(label, labelW - 8, y + cellH / 2 + 3);

    for (let j = 0; j < data.signals.length; j++) {
      const v = data.values[i][j];
      const x = labelW + j * cellW;
      const r = parseInt(SIGNAL_COLORS[0].slice(1, 3), 16);
      const g = parseInt(SIGNAL_COLORS[0].slice(3, 5), 16);
      const b = parseInt(SIGNAL_COLORS[0].slice(5, 7), 16);
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.15 + v * 0.85})`;
      ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
      ctx.fillStyle = v > 0.5 ? "#111827" : "#e5e7eb";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(v.toFixed(2), x + cellW / 2, y + cellH / 2 + 3);
    }
  }
}

import type { SweepSignalData } from "../reports/blocks/SweepBlock.js";

function renderSweepCanvas(
  canvas: HTMLCanvasElement,
  data: SweepSignalData[],
) {
  if (data.length === 0) return;
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

  ctx.font = "11px monospace";
  ctx.fillStyle = "#9ca3af";
  ctx.textAlign = "center";
  const propW = chartW / TEXT_PROPS.length;
  for (let j = 0; j < TEXT_PROPS.length; j++) {
    ctx.fillText(PROP_LABELS[TEXT_PROPS[j]] ?? TEXT_PROPS[j], labelW + j * propW + propW / 2, HEADER_H - 6);
  }

  for (let i = 0; i < data.length; i++) {
    const y = HEADER_H + i * ROW_H;
    const color = SIGNAL_COLORS[0];

    ctx.fillStyle = color;
    ctx.font = "11px monospace";
    ctx.textAlign = "right";
    ctx.fillText(data[i].signal, labelW - 8, y + ROW_H / 2 + 4);

    for (let j = 0; j < data[i].properties.length; j++) {
      const prop = data[i].properties[j];
      const px = labelW + j * propW;
      const barY = y + 8;
      const maxVal = Math.max(Math.abs(prop.low), Math.abs(prop.high), 1);

      const lowW = (Math.abs(prop.low) / maxVal) * (propW / 2 - 8);
      ctx.fillStyle = `${color}66`;
      ctx.fillRect(px + 4, barY, lowW, barH);
      ctx.fillStyle = "#9ca3af";
      ctx.font = "9px monospace";
      ctx.textAlign = "left";
      ctx.fillText(
        typeof prop.low === "number" && prop.low % 1 !== 0 ? prop.low.toFixed(2) : String(prop.low),
        px + 4 + lowW + 2, barY + barH - 3,
      );

      const highY = barY + barH + barGap;
      const highW = (Math.abs(prop.high) / maxVal) * (propW / 2 - 8);
      ctx.fillStyle = color;
      ctx.fillRect(px + 4, highY, highW, barH);
      ctx.fillStyle = "#e5e7eb";
      ctx.font = "9px monospace";
      ctx.textAlign = "left";
      ctx.fillText(
        typeof prop.high === "number" && prop.high % 1 !== 0 ? prop.high.toFixed(2) : String(prop.high),
        px + 4 + highW + 2, highY + barH - 3,
      );

      const deltaY = highY + barH + 12;
      const sign = prop.delta >= 0 ? "+" : "";
      const deltaStr = typeof prop.delta === "number" && prop.delta % 1 !== 0
        ? `${sign}${prop.delta.toFixed(2)}`
        : `${sign}${prop.delta}`;
      ctx.fillStyle = prop.delta >= 0 ? "#4ade80" : "#f87171";
      ctx.font = "10px monospace";
      ctx.textAlign = "left";
      ctx.fillText(`Δ ${deltaStr}`, px + 4, deltaY);
    }
  }
}
