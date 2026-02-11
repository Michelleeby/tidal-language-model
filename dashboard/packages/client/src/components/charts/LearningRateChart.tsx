import { useMemo, useRef, useState } from "react";
import UPlotChart from "./UPlotChart.js";
import ZoomResetButton from "./ZoomResetButton.js";
import type { UPlotChartHandle } from "./UPlotChart.js";
import type { MetricPoint } from "@tidal/shared";

interface LearningRateChartProps {
  points: MetricPoint[];
}

export default function LearningRateChart({ points }: LearningRateChartProps) {
  const chartRef = useRef<UPlotChartHandle>(null);
  const [zoomed, setZoomed] = useState(false);

  const data = useMemo(() => {
    if (points.length === 0) return [new Float64Array(0), new Float64Array(0)] as const;
    const steps = new Float64Array(points.map((p) => p.step));
    const lr = new Float64Array(
      points.map((p) => (p["Learning Rate"] as number) ?? 0),
    );
    return [steps, lr] as const;
  }, [points]);

  if (points.length === 0) {
    return <div className="text-gray-500 text-sm p-4">No LR data yet</div>;
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4 relative">
      <h3 className="text-sm font-medium text-gray-300 mb-2">Learning Rate</h3>
      <ZoomResetButton visible={zoomed} onReset={() => chartRef.current?.resetZoom()} />
      <UPlotChart
        ref={chartRef}
        data={[data[0], data[1]]}
        onZoomChange={setZoomed}
        options={{
          title: "",
          scales: {
            x: { time: false },
          },
          axes: [
            { label: "Step", stroke: "#9ca3af", grid: { stroke: "#1f2937" } },
            {
              label: "LR",
              stroke: "#9ca3af",
              grid: { stroke: "#1f2937" },
              values: (_u: unknown, vals: number[]) => vals.map((v) => v.toExponential(1)),
            },
          ],
          series: [
            {},
            { label: "Learning Rate", stroke: "#10b981", width: 1.5 },
          ],
        }}
        height={200}
      />
    </div>
  );
}
