import { useMemo, useRef, useState, type ReactNode } from "react";
import UPlotChart from "./UPlotChart.js";
import ZoomResetButton from "./ZoomResetButton.js";
import type { UPlotChartHandle } from "./UPlotChart.js";
import type { RLTrainingHistory } from "@tidal/shared";

interface RLGateSignalsChartProps {
  history: RLTrainingHistory | null;
  actions?: ReactNode;
}

export default function RLGateSignalsChart({
  history,
  actions,
}: RLGateSignalsChartProps) {
  const chartRef = useRef<UPlotChartHandle>(null);
  const [zoomed, setZoomed] = useState(false);

  const data = useMemo(() => {
    const modulation = history?.gate_modulation ?? [];
    const len = modulation.length;
    if (len === 0)
      return [
        new Float64Array(0),
        new Float64Array(0),
      ] as const;
    const steps = new Float64Array(modulation.map((_, i) => i));
    return [
      steps,
      new Float64Array(modulation),
    ] as const;
  }, [history]);

  if (!history?.gate_modulation?.length) {
    return (
      <div className="text-gray-500 text-sm p-4">
        No gate signal data yet
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4 relative">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-300">Gate Signals</h3>
        {actions}
      </div>
      <ZoomResetButton
        visible={zoomed}
        onReset={() => chartRef.current?.resetZoom()}
      />
      <UPlotChart
        ref={chartRef}
        data={[data[0], data[1]]}
        onZoomChange={setZoomed}
        options={{
          title: "",
          scales: { x: { time: false }, y: { min: 0, max: 1 } },
          axes: [
            {
              label: "Iteration",
              stroke: "#9ca3af",
              grid: { stroke: "#1f2937" },
            },
            {
              label: "Signal",
              stroke: "#9ca3af",
              grid: { stroke: "#1f2937" },
            },
          ],
          series: [
            {},
            { label: "Modulation", stroke: "#a78bfa", width: 1.5 },
          ],
        }}
      />
    </div>
  );
}
