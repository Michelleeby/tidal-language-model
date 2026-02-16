import { useMemo, useRef, useState, type ReactNode } from "react";
import UPlotChart from "./UPlotChart.js";
import ZoomResetButton from "./ZoomResetButton.js";
import type { UPlotChartHandle } from "./UPlotChart.js";
import type { RLTrainingHistory } from "@tidal/shared";

interface RLRewardComponentsChartProps {
  history: RLTrainingHistory | null;
  actions?: ReactNode;
}

export default function RLRewardComponentsChart({
  history,
  actions,
}: RLRewardComponentsChartProps) {
  const chartRef = useRef<UPlotChartHandle>(null);
  const [zoomed, setZoomed] = useState(false);

  const data = useMemo(() => {
    const perplexity = history?.reward_perplexity ?? [];
    const diversity = history?.reward_diversity ?? [];
    const repetition = history?.reward_repetition ?? [];
    const coherence = history?.reward_coherence ?? [];
    const len = perplexity.length;
    if (len === 0)
      return [
        new Float64Array(0),
        new Float64Array(0),
        new Float64Array(0),
        new Float64Array(0),
        new Float64Array(0),
      ] as const;
    const steps = new Float64Array(perplexity.map((_, i) => i));
    return [
      steps,
      new Float64Array(perplexity),
      new Float64Array(diversity),
      new Float64Array(repetition),
      new Float64Array(coherence),
    ] as const;
  }, [history]);

  if (!history?.reward_perplexity?.length) {
    return (
      <div className="text-gray-500 text-sm p-4">
        No reward component data yet
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4 relative">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-300">Reward Components</h3>
        {actions}
      </div>
      <ZoomResetButton
        visible={zoomed}
        onReset={() => chartRef.current?.resetZoom()}
      />
      <UPlotChart
        ref={chartRef}
        data={[data[0], data[1], data[2], data[3], data[4]]}
        onZoomChange={setZoomed}
        options={{
          title: "",
          scales: { x: { time: false } },
          axes: [
            {
              label: "Iteration",
              stroke: "#9ca3af",
              grid: { stroke: "#1f2937" },
            },
            {
              label: "Reward",
              stroke: "#9ca3af",
              grid: { stroke: "#1f2937" },
            },
          ],
          series: [
            {},
            { label: "Perplexity", stroke: "#c084fc", width: 1.5 },
            { label: "Diversity", stroke: "#4ade80", width: 1.5 },
            { label: "Repetition", stroke: "#fb923c", width: 1.5 },
            { label: "Coherence", stroke: "#38bdf8", width: 1.5 },
          ],
        }}
      />
    </div>
  );
}
