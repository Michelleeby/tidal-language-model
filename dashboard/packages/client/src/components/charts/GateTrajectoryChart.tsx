import { useMemo, useRef, useState } from "react";
import UPlotChart from "./UPlotChart.js";
import ZoomResetButton from "./ZoomResetButton.js";
import type { UPlotChartHandle } from "./UPlotChart.js";
import type { GenerationTrajectory, GateEffectsStep } from "@tidal/shared";

type TrajectoryView = "signals" | "effects";

/**
 * Pure function to build uPlot-compatible chart data from a trajectory.
 * Exported for testing.
 */
export function buildTrajectoryChartData(
  trajectory: GenerationTrajectory | null | undefined,
  view: TrajectoryView,
): Float64Array[] {
  if (!trajectory || trajectory.gateSignals.length === 0) {
    if (view === "signals") {
      return [
        new Float64Array(0),
        new Float64Array(0),
      ];
    }
    return [
      new Float64Array(0),
      new Float64Array(0),
      new Float64Array(0),
      new Float64Array(0),
      new Float64Array(0),
    ];
  }

  const len = trajectory.gateSignals.length;
  const steps = new Float64Array(Array.from({ length: len }, (_, i) => i));

  if (view === "signals") {
    return [
      steps,
      new Float64Array(trajectory.gateSignals.map((s: number[]) => s[0])),
    ];
  }

  // effects view
  return [
    steps,
    new Float64Array(trajectory.effects.map((e: GateEffectsStep) => e.temperature)),
    new Float64Array(trajectory.effects.map((e: GateEffectsStep) => e.repetition_penalty)),
    new Float64Array(trajectory.effects.map((e: GateEffectsStep) => e.top_k)),
    new Float64Array(trajectory.effects.map((e: GateEffectsStep) => e.top_p)),
  ];
}

interface GateTrajectoryChartProps {
  trajectory: GenerationTrajectory;
}

export default function GateTrajectoryChart({
  trajectory,
}: GateTrajectoryChartProps) {
  const chartRef = useRef<UPlotChartHandle>(null);
  const [zoomed, setZoomed] = useState(false);
  const [view, setView] = useState<TrajectoryView>("signals");

  const data = useMemo(
    () => buildTrajectoryChartData(trajectory, view),
    [trajectory, view],
  );

  const signalsOptions = useMemo(
    () => ({
      title: "",
      scales: { x: { time: false }, y: { min: 0, max: 1 } },
      axes: [
        {
          label: "Step",
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
    }),
    [],
  );

  const effectsOptions = useMemo(
    () => ({
      title: "",
      scales: { x: { time: false } },
      axes: [
        {
          label: "Step",
          stroke: "#9ca3af",
          grid: { stroke: "#1f2937" },
        },
        {
          label: "Value",
          stroke: "#9ca3af",
          grid: { stroke: "#1f2937" },
        },
      ],
      series: [
        {},
        { label: "Temperature", stroke: "#fb923c", width: 1.5 },
        { label: "Rep. Penalty", stroke: "#a78bfa", width: 1.5 },
        { label: "Top-K", stroke: "#38bdf8", width: 1.5 },
        { label: "Top-P", stroke: "#fbbf24", width: 1.5 },
      ],
    }),
    [],
  );

  return (
    <div className="bg-gray-900 rounded-lg p-4 relative">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-300">
          Gate Trajectory
        </h3>
        <div className="flex gap-1">
          <button
            className={`px-2 py-0.5 text-xs rounded ${
              view === "signals"
                ? "bg-gray-700 text-gray-200"
                : "text-gray-500 hover:text-gray-300"
            }`}
            onClick={() => setView("signals")}
          >
            Signals
          </button>
          <button
            className={`px-2 py-0.5 text-xs rounded ${
              view === "effects"
                ? "bg-gray-700 text-gray-200"
                : "text-gray-500 hover:text-gray-300"
            }`}
            onClick={() => setView("effects")}
          >
            Effects
          </button>
        </div>
      </div>
      <ZoomResetButton
        visible={zoomed}
        onReset={() => chartRef.current?.resetZoom()}
      />
      <UPlotChart
        ref={chartRef}
        data={data}
        onZoomChange={setZoomed}
        height={200}
        options={view === "signals" ? signalsOptions : effectsOptions}
      />
      {/* Token text strip */}
      <div className="mt-2 flex flex-wrap gap-0.5 max-h-16 overflow-y-auto font-mono text-xs text-gray-400">
        {trajectory.tokenTexts.map((t: string, i: number) => (
          <span
            key={i}
            className="bg-gray-800 px-1 rounded"
            title={`Step ${i}`}
          >
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}
