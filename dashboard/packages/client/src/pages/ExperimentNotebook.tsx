import { useMemo, useState } from "react";
import { useExperiments } from "../hooks/useExperiments.js";
import {
  useFullMetrics,
  useRLMetrics,
  useCheckpoints,
  useEvaluation,
  useAblation,
} from "../hooks/useMetrics.js";
import { useSSE } from "../hooks/useSSE.js";
import { useJobSSE } from "../hooks/useJobSSE.js";
import { useExperimentStore } from "../stores/experimentStore.js";
import CollapsibleSection from "../components/notebook/CollapsibleSection.js";
import LossCurves from "../components/charts/LossCurves.js";
import LearningRateChart from "../components/charts/LearningRateChart.js";
import PerplexityChart from "../components/charts/PerplexityChart.js";
import ThroughputChart from "../components/charts/ThroughputChart.js";
import RLRewardCurve from "../components/charts/RLRewardCurve.js";
import RLLossChart from "../components/charts/RLLossChart.js";
import RLEpisodeLengthChart from "../components/charts/RLEpisodeLengthChart.js";
import AblationComparison from "../components/charts/AblationComparison.js";
import CheckpointBrowser from "../components/charts/CheckpointBrowser.js";
import TrainingStatusCard from "../components/status/TrainingStatusCard.js";
import MetricCards from "../components/status/MetricCards.js";
import MetricCarousel from "../components/status/MetricCarousel.js";
import SamplePreviews from "../components/samples/SamplePreviews.js";
import TrainingControlBar from "../components/jobs/TrainingControlBar.js";
import RLTrainingTrigger from "../components/jobs/RLTrainingTrigger.js";
import LogViewer from "../components/logs/LogViewer.js";
import ConfigViewer from "../components/config/ConfigViewer.js";
import GenerationSection from "../components/generation/GenerationSection.js";
import ChartExportButton from "../components/charts/ChartExportButton.js";
import { lttbDownsample } from "../utils/downsample.js";
import { useReportData } from "../hooks/useReportData.js";
import { generateHTMLReport, generateMarkdownReport } from "../utils/report.js";
import type { MetricPoint, TrainingStatus } from "@tidal/shared";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client.js";
import { useActiveJob } from "../hooks/useJobs.js";

const CHART_SYNC_KEY = "training-charts";
const CLIENT_MAX_POINTS = 8000;

export default function ExperimentNotebook() {
  const { selectedExpId, setSelectedExpId } = useExperimentStore();
  const [reportMenuOpen, setReportMenuOpen] = useState(false);

  const { data: expData, isLoading: expLoading } = useExperiments();
  const { data: metricsData } = useFullMetrics(selectedExpId);
  const { data: rlData } = useRLMetrics(selectedExpId);
  const { data: checkpointsData } = useCheckpoints(selectedExpId);
  const { data: evalData } = useEvaluation(selectedExpId);
  const { data: ablationData } = useAblation(selectedExpId);
  const { data: activeJobData } = useActiveJob();
  const activeJobId = activeJobData?.job?.jobId;

  useSSE(selectedExpId);
  useJobSSE();

  const { data: statusData } = useQuery({
    queryKey: ["status", selectedExpId],
    queryFn: async () => {
      const res = await api.getStatus(selectedExpId!);
      return res.status;
    },
    enabled: !!selectedExpId,
    staleTime: Infinity,
  });
  const status: TrainingStatus | null = statusData ?? null;

  // Merge historical data with live SSE data
  const queryClient = useQueryClient();
  const livePoints =
    queryClient.getQueryData<MetricPoint[]>([
      "metrics",
      selectedExpId,
      "live",
    ]) ?? [];
  const historicalPoints = metricsData?.points ?? [];

  const points = useMemo(() => {
    if (historicalPoints.length === 0 && livePoints.length === 0) return [];
    const lastHistoricalStep =
      historicalPoints.length > 0
        ? historicalPoints[historicalPoints.length - 1].step
        : -1;
    const newLivePoints = livePoints.filter(
      (p) => p.step > lastHistoricalStep,
    );
    const merged =
      newLivePoints.length > 0
        ? [...historicalPoints, ...newLivePoints]
        : historicalPoints;
    if (merged.length > CLIENT_MAX_POINTS) {
      return lttbDownsample(merged, CLIENT_MAX_POINTS);
    }
    return merged;
  }, [historicalPoints, livePoints]);

  const latestPoint = points.length > 0 ? points[points.length - 1] : null;

  // Auto-select first experiment if none selected
  if (!selectedExpId && expData?.experiments?.[0]) {
    setSelectedExpId(expData.experiments[0].id);
  }

  const rlHistory = rlData?.metrics?.history ?? null;
  const hasRLData = rlHistory && rlHistory.episode_rewards.length > 0;

  // Report data assembly
  const reportData = useReportData(selectedExpId);

  function handleExportReport(format: "html" | "markdown") {
    setReportMenuOpen(false);
    if (!reportData) return;

    let content: string;
    let filename: string;
    let mimeType: string;

    if (format === "html") {
      content = generateHTMLReport(reportData);
      filename = `report-${selectedExpId}.html`;
      mimeType = "text/html";
    } else {
      content = generateMarkdownReport(reportData);
      filename = `report-${selectedExpId}.md`;
      mimeType = "text/markdown";
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  if (!selectedExpId) {
    return (
      <div className="flex flex-col items-center justify-center py-20 space-y-4">
        <div className="text-gray-500 text-sm">
          {expLoading
            ? "Loading experiments..."
            : "Select an experiment from the sidebar to begin."}
        </div>
        <TrainingControlBar />
      </div>
    );
  }

  return (
    <div className="space-y-3 max-w-6xl">
      {/* Experiment header */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div>
          <h2 className="text-lg font-semibold font-mono text-gray-100">
            {selectedExpId}
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <TrainingControlBar />
          <div className="relative">
            <button
              onClick={() => setReportMenuOpen((v) => !v)}
              className="px-3 py-1.5 text-sm rounded bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors border border-gray-700"
            >
              Export Report
            </button>
            {reportMenuOpen && (
              <div className="absolute right-0 mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-lg z-10 py-1 min-w-[160px]">
                <button
                  onClick={() => handleExportReport("html")}
                  className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700"
                >
                  HTML Report
                </button>
                <button
                  onClick={() => handleExportReport("markdown")}
                  className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700"
                >
                  Markdown Report
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 1. Overview */}
      <CollapsibleSection title="Overview" defaultOpen>
        <MetricCarousel>
          <MetricCards latest={latestPoint} />
          <TrainingStatusCard status={status} />
        </MetricCarousel>
      </CollapsibleSection>

      {/* 2. Monitor */}
      <CollapsibleSection title="Monitor" defaultOpen>
        <div className="space-y-6">
          {/* Language Model Training */}
          <div className="space-y-4">
            <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Language Model Training
            </h4>
            <LossCurves
              points={points}
              syncKey={CHART_SYNC_KEY}
              actions={
                <ChartExportButton
                  data={{
                    headers: ["step", "loss", "smoothed_loss"],
                    rows: points.map((p) => [
                      p.step,
                      (p["Losses/Total"] as number) ?? 0,
                      (p["Losses/Total"] as number) ?? 0,
                    ]),
                  }}
                  filename="loss_curves"
                />
              }
            />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <LearningRateChart
                points={points}
                syncKey={CHART_SYNC_KEY}
                actions={
                  <ChartExportButton
                    data={{
                      headers: ["step", "learning_rate"],
                      rows: points.map((p) => [
                        p.step,
                        (p["Learning Rate"] as number) ?? 0,
                      ]),
                    }}
                    filename="learning_rate"
                  />
                }
              />
              <PerplexityChart
                points={points}
                syncKey={CHART_SYNC_KEY}
                actions={
                  <ChartExportButton
                    data={{
                      headers: ["step", "perplexity"],
                      rows: points.map((p) => [
                        p.step,
                        Math.exp((p["Losses/Total"] as number) ?? 0),
                      ]),
                    }}
                    filename="perplexity"
                  />
                }
              />
            </div>
            <ThroughputChart
              points={points}
              syncKey={CHART_SYNC_KEY}
              actions={
                <ChartExportButton
                  data={{
                    headers: ["step", "iterations_per_second"],
                    rows: points
                      .filter((p) => p["Iterations/Second"] != null)
                      .map((p) => [
                        p.step,
                        (p["Iterations/Second"] as number) ?? 0,
                      ]),
                  }}
                  filename="throughput"
                />
              }
            />
          </div>

          {/* RL Gating */}
          <div className="space-y-4">
            <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              RL Gating
            </h4>
            <RLTrainingTrigger selectedExpId={selectedExpId} />
            {hasRLData && (
              <>
                <RLRewardCurve
                  history={rlHistory}
                  actions={
                    <ChartExportButton
                      data={{
                        headers: ["episode", "mean_reward"],
                        rows: rlHistory!.episode_rewards.map((r, i) => [i, r]),
                      }}
                      filename="rl_rewards"
                    />
                  }
                />
                <RLLossChart
                  history={rlHistory}
                  actions={
                    <ChartExportButton
                      data={{
                        headers: [
                          "step",
                          "policy_loss",
                          "value_loss",
                          "entropy",
                        ],
                        rows: rlHistory!.policy_loss.map((_, i) => [
                          i,
                          rlHistory!.policy_loss[i],
                          rlHistory!.value_loss[i],
                          rlHistory!.entropy[i],
                        ]),
                      }}
                      filename="rl_losses"
                    />
                  }
                />
                <RLEpisodeLengthChart
                  history={rlHistory}
                  actions={
                    <ChartExportButton
                      data={{
                        headers: ["episode", "mean_length"],
                        rows: rlHistory!.episode_lengths.map((l, i) => [i, l]),
                      }}
                      filename="rl_episode_lengths"
                    />
                  }
                />
              </>
            )}
            <AblationComparison results={ablationData?.results ?? null} />
          </div>
        </div>
      </CollapsibleSection>

      {/* 3. Configuration */}
      <CollapsibleSection title="Configuration">
        <ConfigViewer />
      </CollapsibleSection>

      {/* 4. Checkpoints */}
      <CollapsibleSection title="Checkpoints">
        <CheckpointBrowser checkpoints={checkpointsData?.checkpoints ?? []} />
      </CollapsibleSection>

      {/* 5. Generation */}
      <CollapsibleSection title="Generation">
        <GenerationSection expId={selectedExpId} />
      </CollapsibleSection>

      {/* 6. Logs */}
      <CollapsibleSection title="Logs">
        <LogViewer jobId={activeJobId} />
      </CollapsibleSection>

      {/* 7. Samples */}
      <CollapsibleSection title="Samples">
        <SamplePreviews results={evalData?.results ?? null} />
      </CollapsibleSection>
    </div>
  );
}
