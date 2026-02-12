import { useRef, useEffect } from "react";
import type { AblationResults } from "@tidal/shared";

interface AblationComparisonProps {
  results: AblationResults | null;
}

export default function AblationComparison({
  results,
}: AblationComparisonProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!results || !containerRef.current) return;

    // Plotly is loaded via CDN script tag (too large to bundle)
    const Plotly = window.Plotly;
    if (!Plotly) return;

    const policies = Object.keys(results);
    const metrics = ["mean_reward", "mean_diversity", "mean_perplexity"];
    const colors = ["#3b82f6", "#22c55e", "#a855f7"];

    const traces = metrics.map((metric, i) => ({
      x: policies,
      y: policies.map((p) => results[p][metric as keyof typeof results[string]]),
      name: metric.replace("mean_", ""),
      type: "bar" as const,
      marker: { color: colors[i] },
    }));

    Plotly.newPlot(containerRef.current!, traces, {
      barmode: "group",
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: "#9ca3af" },
      xaxis: { title: "Policy" },
      yaxis: { title: "Value", gridcolor: "#1f2937" },
      margin: { t: 20, b: 60, l: 60, r: 20 },
      legend: { orientation: "h", y: -0.2 },
    }, { responsive: true });
  }, [results]);

  if (!results) {
    return (
      <div className="text-gray-500 text-sm p-4">No ablation data</div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-2">
        Ablation Comparison
      </h3>
      <div ref={containerRef} style={{ height: 300 }} />
    </div>
  );
}
