import { useMemo, useRef, useState, type ReactNode } from "react";
import UPlotChart from "./UPlotChart.js";
import ZoomResetButton from "./ZoomResetButton.js";
import type { UPlotChartHandle } from "./UPlotChart.js";
import type { RLTrainingHistory } from "@tidal/shared";

interface RLRewardCurveProps {
  history: RLTrainingHistory | null;
  actions?: ReactNode;
}

export default function RLRewardCurve({ history, actions }: RLRewardCurveProps) {
  const chartRef = useRef<UPlotChartHandle>(null);
  const [zoomed, setZoomed] = useState(false);

  const data = useMemo(() => {
    if (!history || history.episode_rewards.length === 0)
      return [new Float64Array(0), new Float64Array(0)] as const;
    const steps = new Float64Array(
      history.episode_rewards.map((_, i) => i),
    );
    const rewards = new Float64Array(history.episode_rewards);
    return [steps, rewards] as const;
  }, [history]);

  if (!history || history.episode_rewards.length === 0) {
    return <div className="text-gray-500 text-sm p-4">No RL data yet</div>;
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4 relative">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-300">Episode Rewards</h3>
        {actions}
      </div>
      <ZoomResetButton visible={zoomed} onReset={() => chartRef.current?.resetZoom()} />
      <UPlotChart
        ref={chartRef}
        data={[data[0], data[1]]}
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
            { label: "Mean Reward", stroke: "#22c55e", width: 1.5 },
          ],
        }}
      />
    </div>
  );
}
