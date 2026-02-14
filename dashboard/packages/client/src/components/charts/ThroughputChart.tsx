import { useMemo, useRef, useState, type ReactNode } from "react";
import UPlotChart from "./UPlotChart.js";
import ZoomResetButton from "./ZoomResetButton.js";
import type { UPlotChartHandle } from "./UPlotChart.js";
import type { MetricPoint } from "@tidal/shared";

interface ThroughputChartProps {
  points: MetricPoint[];
  syncKey?: string;
  actions?: ReactNode;
}

export default function ThroughputChart({ points, syncKey, actions }: ThroughputChartProps) {
  const chartRef = useRef<UPlotChartHandle>(null);
  const [zoomed, setZoomed] = useState(false);

  const filteredPoints = useMemo(
    () => points.filter((p) => p["Iterations/Second"] != null),
    [points],
  );

  const data = useMemo(() => {
    if (filteredPoints.length === 0)
      return [new Float64Array(0), new Float64Array(0)] as const;
    const steps = new Float64Array(filteredPoints.map((p) => p.step));
    const throughput = new Float64Array(
      filteredPoints.map((p) => (p["Iterations/Second"] as number) ?? 0),
    );
    return [steps, throughput] as const;
  }, [filteredPoints]);

  if (filteredPoints.length === 0) {
    return (
      <div className="text-gray-500 text-sm p-4">No throughput data yet</div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4 relative">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-300">Throughput</h3>
        {actions}
      </div>
      <ZoomResetButton
        visible={zoomed}
        onReset={() => chartRef.current?.resetZoom()}
      />
      <UPlotChart
        ref={chartRef}
        data={[data[0], data[1]]}
        syncKey={syncKey}
        onZoomChange={setZoomed}
        height={200}
        options={{
          title: "",
          scales: { x: { time: false } },
          axes: [
            {
              label: "Step",
              stroke: "#9ca3af",
              grid: { stroke: "#1f2937" },
            },
            {
              label: "Iterations/s",
              stroke: "#9ca3af",
              grid: { stroke: "#1f2937" },
            },
          ],
          series: [
            {},
            { label: "Iterations/s", stroke: "#f59e0b", width: 1.5 },
          ],
        }}
      />
    </div>
  );
}
