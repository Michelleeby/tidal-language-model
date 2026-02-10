import { useMemo } from "react";
import UPlotChart from "./UPlotChart.js";
import type { MetricPoint } from "@tidal/shared";

interface PerplexityChartProps {
  points: MetricPoint[];
}

export default function PerplexityChart({ points }: PerplexityChartProps) {
  const data = useMemo(() => {
    if (points.length === 0) return [new Float64Array(0), new Float64Array(0)] as const;
    const steps = new Float64Array(points.map((p) => p.step));
    const perplexity = new Float64Array(
      points.map((p) => {
        const loss = (p["Losses/Total"] as number) ?? 0;
        return Math.exp(loss);
      }),
    );
    return [steps, perplexity] as const;
  }, [points]);

  if (points.length === 0) {
    return <div className="text-gray-500 text-sm p-4">No data yet</div>;
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-2">
        Perplexity (exp(loss))
      </h3>
      <UPlotChart
        data={[data[0], data[1]]}
        options={{
          title: "",
          scales: {
            x: { time: false },
          },
          axes: [
            { label: "Step", stroke: "#9ca3af", grid: { stroke: "#1f2937" } },
            {
              label: "Perplexity",
              stroke: "#9ca3af",
              grid: { stroke: "#1f2937" },
            },
          ],
          series: [
            {},
            { label: "Perplexity", stroke: "#a855f7", width: 1.5 },
          ],
        }}
        height={200}
      />
    </div>
  );
}
