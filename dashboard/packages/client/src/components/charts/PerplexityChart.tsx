import { useMemo, useRef, useState, type ReactNode } from "react";
import UPlotChart from "./UPlotChart.js";
import ZoomResetButton from "./ZoomResetButton.js";
import type { UPlotChartHandle } from "./UPlotChart.js";
import type { MetricPoint } from "@tidal/shared";

interface PerplexityChartProps {
  points: MetricPoint[];
  syncKey?: string;
  actions?: ReactNode;
}

export default function PerplexityChart({ points, syncKey, actions }: PerplexityChartProps) {
  const chartRef = useRef<UPlotChartHandle>(null);
  const [zoomed, setZoomed] = useState(false);

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
    <div className="bg-gray-900 rounded-lg p-4 relative">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-300">
          Perplexity (exp(loss))
        </h3>
        {actions}
      </div>
      <ZoomResetButton visible={zoomed} onReset={() => chartRef.current?.resetZoom()} />
      <UPlotChart
        ref={chartRef}
        data={[data[0], data[1]]}
        syncKey={syncKey}
        onZoomChange={setZoomed}
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
