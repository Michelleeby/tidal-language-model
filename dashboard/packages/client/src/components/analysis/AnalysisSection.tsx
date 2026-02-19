import { useState, useEffect, useRef } from "react";
import { useCheckpoints } from "../../hooks/useMetrics.js";
import { useAnalyses, useSaveAnalysis, useDeleteAnalysis } from "../../hooks/useAnalyses.js";
import { useTrajectoryAnalysis } from "../../hooks/useTrajectoryAnalysis.js";
import { api } from "../../api/client.js";
import GateTrajectoryChart from "../charts/GateTrajectoryChart.js";
import {
  extractHeatmapRenderData,
  extractVarianceSummary,
  extractSweepRenderData,
  extractInterpretabilitySummary,
  renderHeatmap,
  renderSweepPanel,
  SIGNAL_COLORS,
  HEATMAP_DIMENSIONS,
  SWEEP_DIMENSIONS,
  TEXT_PROP_LABELS,
} from "../../utils/chartRenderers.js";
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

function CrossPromptTab({ expId }: { expId: string }) {
  const { data: checkpointsData } = useCheckpoints(expId);
  const saveAnalysis = useSaveAnalysis();
  const analysis = useTrajectoryAnalysis();
  const [gatingMode, setGatingMode] = useState<"fixed" | "random" | "learned">("fixed");
  const [cachedBatch, setCachedBatch] = useState<BatchAnalysis | null>(null);
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

  // Display live data or cached data
  const batch = analysis.data?.batchAnalysis ?? cachedBatch;

  // Re-render canvas whenever batch data changes (runs after DOM commit)
  useEffect(() => {
    if (!batch || !heatmapCanvasRef.current) return;
    const data = extractHeatmapRenderData(batch);
    renderHeatmap(heatmapCanvasRef.current, data);
  }, [batch]);

  const varianceSummary = batch ? extractVarianceSummary(batch.crossPromptVariance) : [];

  const handleSelectCached = async (id: string) => {
    const resp = await api.getAnalysis(id);
    const cachedData = resp.analysis.data as unknown as AnalyzeResponse;
    if (cachedData.batchAnalysis) {
      // Reset live mutation so it doesn't shadow the cached data
      analysis.reset();
      // Store in state — triggers re-render → canvas mounts → useEffect renders
      setCachedBatch(cachedData.batchAnalysis);
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
              width={HEATMAP_DIMENSIONS.width}
              height={HEATMAP_DIMENSIONS.height}
              className="w-full rounded bg-gray-950"
              style={{ height: HEATMAP_DIMENSIONS.height }}
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

function SweepTab({ expId }: { expId: string }) {
  const { data: checkpointsData } = useCheckpoints(expId);
  const saveAnalysis = useSaveAnalysis();
  const analysis = useTrajectoryAnalysis();
  const [prompt, setPrompt] = useState("Once upon a time,");
  const [cachedSweep, setCachedSweep] = useState<SweepAnalysis | null>(null);
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

  // Display live data or cached data
  const sweep = analysis.data?.sweepAnalysis ?? cachedSweep;

  // Re-render canvas whenever sweep data changes (runs after DOM commit)
  useEffect(() => {
    if (!sweep || !canvasRef.current) return;
    const panelData = extractSweepRenderData(sweep);
    renderSweepPanel(canvasRef.current, panelData);
  }, [sweep]);

  const summaryLines = sweep
    ? extractInterpretabilitySummary(sweep.interpretabilityMap)
    : [];
  const panelData = sweep ? extractSweepRenderData(sweep) : [];
  const panelH = panelData.length > 0 ? SWEEP_DIMENSIONS.headerHeight + panelData.length * SWEEP_DIMENSIONS.rowHeight : 200;

  const handleSelectCached = async (id: string) => {
    const resp = await api.getAnalysis(id);
    const cachedData = resp.analysis.data as unknown as AnalyzeResponse;
    if (cachedData.sweepAnalysis) {
      // Reset live mutation so it doesn't shadow the cached data
      analysis.reset();
      // Store in state — triggers re-render → canvas mounts → useEffect renders
      setCachedSweep(cachedData.sweepAnalysis);
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
              width={SWEEP_DIMENSIONS.width}
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
