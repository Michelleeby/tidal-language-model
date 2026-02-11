import { useMemo, useRef, useState } from "react";
import UPlotChart from "./UPlotChart.js";
import ZoomResetButton from "./ZoomResetButton.js";
import type { UPlotChartHandle } from "./UPlotChart.js";
import type { MetricPoint } from "@tidal/shared";

interface LossCurvesProps {
  points: MetricPoint[];
}

export default function LossCurves({ points }: LossCurvesProps) {
  const chartRef = useRef<UPlotChartHandle>(null);
  const [zoomed, setZoomed] = useState(false);

  const data = useMemo(() => {
    if (points.length === 0) return [new Float64Array(0), new Float64Array(0)] as const;
    const steps = new Float64Array(points.map((p) => p.step));
    const loss = new Float64Array(
      points.map((p) => (p["Losses/Total"] as number) ?? 0),
    );
    return [steps, loss] as const;
  }, [points]);

  if (points.length === 0) {
    return <div className="text-gray-500 text-sm p-4">No loss data yet</div>;
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4 relative">
      <h3 className="text-sm font-medium text-gray-300 mb-2">Loss Curve</h3>
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
            { label: "Loss", stroke: "#9ca3af", grid: { stroke: "#1f2937" } },
          ],
          series: [
            {},
            { label: "Loss", stroke: "#3b82f6", width: 1.5 },
          ],
        }}
      />
    </div>
  );
}
