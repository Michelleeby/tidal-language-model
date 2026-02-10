import { useMemo } from "react";
import UPlotChart from "./UPlotChart.js";
import type { RLTrainingHistory } from "@tidal/shared";

interface RLLossChartProps {
  history: RLTrainingHistory | null;
}

export default function RLLossChart({ history }: RLLossChartProps) {
  const data = useMemo(() => {
    if (!history || history.policy_loss.length === 0)
      return [
        new Float64Array(0),
        new Float64Array(0),
        new Float64Array(0),
        new Float64Array(0),
      ] as const;
    const steps = new Float64Array(history.policy_loss.map((_, i) => i));
    return [
      steps,
      new Float64Array(history.policy_loss),
      new Float64Array(history.value_loss),
      new Float64Array(history.entropy),
    ] as const;
  }, [history]);

  if (!history || history.policy_loss.length === 0) {
    return <div className="text-gray-500 text-sm p-4">No RL loss data</div>;
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-2">RL Losses</h3>
      <UPlotChart
        data={[data[0], data[1], data[2], data[3]]}
        options={{
          title: "",
          scales: { x: { time: false } },
          axes: [
            {
              label: "Iteration",
              stroke: "#9ca3af",
              grid: { stroke: "#1f2937" },
            },
            { label: "Value", stroke: "#9ca3af", grid: { stroke: "#1f2937" } },
          ],
          series: [
            {},
            { label: "Policy Loss", stroke: "#ef4444", width: 1.5 },
            { label: "Value Loss", stroke: "#3b82f6", width: 1.5 },
            { label: "Entropy", stroke: "#f59e0b", width: 1.5 },
          ],
        }}
      />
    </div>
  );
}
