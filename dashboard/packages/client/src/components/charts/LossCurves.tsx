import { useMemo, useRef, useState, type ReactNode } from "react";
import UPlotChart from "./UPlotChart.js";
import ZoomResetButton from "./ZoomResetButton.js";
import type { UPlotChartHandle } from "./UPlotChart.js";
import type { MetricPoint } from "@tidal/shared";

interface LossCurvesProps {
  points: MetricPoint[];
  syncKey?: string;
  actions?: ReactNode;
}

/** Exponential moving average with configurable alpha. */
function ema(values: number[], alpha: number): number[] {
  if (values.length === 0) return [];
  const result = new Array<number>(values.length);
  result[0] = values[0];
  for (let i = 1; i < values.length; i++) {
    result[i] = alpha * result[i - 1] + (1 - alpha) * values[i];
  }
  return result;
}

export default function LossCurves({ points, syncKey, actions }: LossCurvesProps) {
  const chartRef = useRef<UPlotChartHandle>(null);
  const [zoomed, setZoomed] = useState(false);

  const data = useMemo(() => {
    if (points.length === 0)
      return [
        new Float64Array(0),
        new Float64Array(0),
        new Float64Array(0),
      ] as const;
    const steps = new Float64Array(points.map((p) => p.step));
    const rawLoss = points.map((p) => (p["Losses/Total"] as number) ?? 0);
    const loss = new Float64Array(rawLoss);
    const smoothed = new Float64Array(ema(rawLoss, 0.98));
    return [steps, loss, smoothed] as const;
  }, [points]);

  if (points.length === 0) {
    return <div className="text-gray-500 text-sm p-4">No loss data yet</div>;
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4 relative">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-300">Loss Curve</h3>
        {actions}
      </div>
      <ZoomResetButton visible={zoomed} onReset={() => chartRef.current?.resetZoom()} />
      <UPlotChart
        ref={chartRef}
        data={[data[0], data[1], data[2]]}
        syncKey={syncKey}
        onZoomChange={setZoomed}
        options={{
          title: "",
          scales: {
            x: { time: false },
          },
          axes: [
            { label: "Step", stroke: "#9ca3af", grid: { stroke: "#1f2937" } },
            { label: "Loss", stroke: "#9ca3af", grid: { stroke: "#1f2937" } },
          ],
          series: [
            {},
            { label: "Loss", stroke: "#3b82f680", width: 1 },
            { label: "Smoothed", stroke: "#3b82f6", width: 2.5 },
          ],
        }}
      />
    </div>
  );
}
