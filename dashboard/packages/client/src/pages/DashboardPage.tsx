import { useMemo } from "react";
import { useExperiments } from "../hooks/useExperiments.js";
import { useFullMetrics, useRLMetrics, useCheckpoints, useEvaluation, useAblation } from "../hooks/useMetrics.js";
import { useSSE } from "../hooks/useSSE.js";
import { useJobSSE } from "../hooks/useJobSSE.js";
import { useExperimentStore } from "../stores/experimentStore.js";
import LossCurves from "../components/charts/LossCurves.js";
import LearningRateChart from "../components/charts/LearningRateChart.js";
import PerplexityChart from "../components/charts/PerplexityChart.js";
import RLRewardCurve from "../components/charts/RLRewardCurve.js";
import RLLossChart from "../components/charts/RLLossChart.js";
import AblationComparison from "../components/charts/AblationComparison.js";
import CheckpointBrowser from "../components/charts/CheckpointBrowser.js";
import TrainingStatusCard from "../components/status/TrainingStatusCard.js";
import MetricCards from "../components/status/MetricCards.js";
import MetricCarousel from "../components/status/MetricCarousel.js";
import SamplePreviews from "../components/samples/SamplePreviews.js";
import TrainingControlBar from "../components/jobs/TrainingControlBar.js";
import RLTrainingTrigger from "../components/jobs/RLTrainingTrigger.js";
import { lttbDownsample } from "../utils/downsample.js";
import type { MetricPoint } from "@tidal/shared";
import { useQueryClient } from "@tanstack/react-query";

type Tab = "training" | "rl-gating" | "comparison" | "checkpoints" | "samples";

const TABS: { key: Tab; label: string }[] = [
  { key: "training", label: "Training" },
  { key: "rl-gating", label: "RL Gating" },
  { key: "comparison", label: "Comparison" },
  { key: "checkpoints", label: "Checkpoints" },
  { key: "samples", label: "Samples" },
];

const CLIENT_MAX_POINTS = 8000;

export default function DashboardPage() {
  const { selectedExpId, setSelectedExpId, activeTab, setActiveTab } =
    useExperimentStore();

  const { data: expData, isLoading: expLoading } = useExperiments();
  const { data: metricsData } = useFullMetrics(selectedExpId);
  const { data: rlData } = useRLMetrics(selectedExpId);
  const { data: checkpointsData } = useCheckpoints(selectedExpId);
  const { data: evalData } = useEvaluation(selectedExpId);
  const { data: ablationData } = useAblation(selectedExpId);

  // SSE for live updates
  useSSE(selectedExpId);
  useJobSSE();

  // Merge historical data with live SSE data
  const queryClient = useQueryClient();
  const livePoints =
    queryClient.getQueryData<MetricPoint[]>(["metrics", selectedExpId, "live"]) ?? [];
  const historicalPoints = metricsData?.points ?? [];

  const points = useMemo(() => {
    if (historicalPoints.length === 0 && livePoints.length === 0) return [];

    // Find the last step in historical data
    const lastHistoricalStep =
      historicalPoints.length > 0
        ? historicalPoints[historicalPoints.length - 1].step
        : -1;

    // Append only live points that are newer than the historical data
    const newLivePoints = livePoints.filter((p) => p.step > lastHistoricalStep);
    const merged = newLivePoints.length > 0
      ? [...historicalPoints, ...newLivePoints]
      : historicalPoints;

    // Safety cap: client-side LTTB if merged data exceeds limit
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

  return (
    <div className="space-y-4">
      {/* Experiment selector */}
      <div className="flex flex-col gap-1">
        <label className="text-sm text-gray-400">Experiment:</label>
        {expLoading ? (
          <span className="text-sm text-gray-500">Loading...</span>
        ) : (
          <select
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-blue-500 max-w-md"
            value={selectedExpId ?? ""}
            onChange={(e) => setSelectedExpId(e.target.value || null)}
          >
            <option value="">Select experiment</option>
            {expData?.experiments.map((exp) => (
              <option key={exp.id} value={exp.id}>
                {exp.id}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-800 pb-px overflow-x-auto">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.key
                ? "border-blue-500 text-blue-400"
                : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "training" && (
        <div className="space-y-4">
          <TrainingControlBar />
          {selectedExpId ? (
            <>
              <MetricCarousel>
                <MetricCards latest={latestPoint} />
                <TrainingStatusCard
                  status={
                    expData?.experiments.find((e) => e.id === selectedExpId)
                      ?.status ?? null
                  }
                />
              </MetricCarousel>
              <LossCurves points={points} />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <LearningRateChart points={points} />
                <PerplexityChart points={points} />
              </div>
            </>
          ) : (
            <div className="text-gray-500 text-sm py-8 text-center">
              No experiments yet. Start a training job above to begin.
            </div>
          )}
        </div>
      )}

      {activeTab === "rl-gating" && (
        <div className="space-y-4">
          {selectedExpId ? (
            <>
              <RLTrainingTrigger selectedExpId={selectedExpId} />
              <RLRewardCurve history={rlData?.metrics?.history ?? null} />
              <RLLossChart history={rlData?.metrics?.history ?? null} />
              <AblationComparison results={ablationData?.results ?? null} />
            </>
          ) : (
            <div className="text-gray-500 text-sm py-8 text-center">
              Select an experiment to view RL gating metrics.
            </div>
          )}
        </div>
      )}

      {activeTab === "comparison" && (
        <div className="text-gray-500 text-sm py-8 text-center">
          Multi-experiment comparison coming in a future update.
          <br />
          Select multiple experiments to overlay loss curves.
        </div>
      )}

      {activeTab === "checkpoints" && (
        selectedExpId ? (
          <CheckpointBrowser
            checkpoints={checkpointsData?.checkpoints ?? []}
          />
        ) : (
          <div className="text-gray-500 text-sm py-8 text-center">
            Select an experiment to browse checkpoints.
          </div>
        )
      )}

      {activeTab === "samples" && (
        selectedExpId ? (
          <SamplePreviews results={evalData?.results ?? null} />
        ) : (
          <div className="text-gray-500 text-sm py-8 text-center">
            Select an experiment to view sample previews.
          </div>
        )
      )}
    </div>
  );
}
