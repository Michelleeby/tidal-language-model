import { createReactBlockSpec } from "@blocknote/react";
import { useReportData } from "../../../hooks/useReportData.js";
import { useExperiments } from "../../../hooks/useExperiments.js";
import { defaultProps } from "@blocknote/core";

export const MetricsTableBlock = createReactBlockSpec(
  {
    type: "metricsTable" as const,
    propSchema: {
      ...defaultProps,
      experimentId: { default: "" },
    },
    content: "none",
  },
  {
    render: ({ block, editor }: any) => {
      const { experimentId } = block.props;
      const { data: expData } = useExperiments();
      const reportData = useReportData(experimentId || null);
      const isEditable = editor.isEditable;

      const experiments = expData?.experiments ?? [];

      return (
        <div className="my-2 rounded-lg border border-gray-700 bg-gray-900 p-4">
          {isEditable && (
            <div className="mb-3">
              <select
                className="rounded bg-gray-800 border border-gray-600 text-gray-200 text-sm px-2 py-1"
                value={experimentId}
                onChange={(e) => {
                  editor.updateBlock(block, {
                    props: { experimentId: e.target.value },
                  });
                }}
              >
                <option value="">Select experiment...</option>
                {experiments.map((exp) => (
                  <option key={exp.id} value={exp.id}>
                    {exp.id.length > 24 ? `${exp.id.slice(0, 24)}...` : exp.id}
                  </option>
                ))}
              </select>
            </div>
          )}

          {!experimentId ? (
            <div className="text-gray-500 text-sm py-4 text-center">
              Select an experiment to display metrics
            </div>
          ) : !reportData ? (
            <div className="text-gray-500 text-sm py-4 text-center">
              Loading metrics...
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700 text-gray-400">
                  <th className="text-left py-2 px-3 font-medium">Metric</th>
                  <th className="text-right py-2 px-3 font-medium">Value</th>
                </tr>
              </thead>
              <tbody className="text-gray-200">
                <MetricRow label="Experiment" value={reportData.experimentId} />
                <MetricRow
                  label="Status"
                  value={reportData.metrics.trainingStatus ?? "N/A"}
                />
                <MetricRow
                  label="Total Steps"
                  value={reportData.metrics.totalSteps?.toLocaleString() ?? "N/A"}
                />
                <MetricRow
                  label="Final Loss"
                  value={reportData.metrics.finalLoss?.toFixed(4) ?? "N/A"}
                />
                <MetricRow
                  label="Final Perplexity"
                  value={reportData.metrics.finalPerplexity?.toFixed(2) ?? "N/A"}
                />
                <MetricRow
                  label="Checkpoints"
                  value={String(reportData.checkpoints.length)}
                />
                {reportData.rlMetrics && (
                  <>
                    <MetricRow
                      label="RL Episodes"
                      value={String(reportData.rlMetrics.episodeCount)}
                    />
                    <MetricRow
                      label="Final Reward"
                      value={reportData.rlMetrics.finalReward?.toFixed(4) ?? "N/A"}
                    />
                  </>
                )}
              </tbody>
            </table>
          )}
        </div>
      );
    },
  },
);

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <tr className="border-b border-gray-800">
      <td className="py-2 px-3 text-gray-400">{label}</td>
      <td className="py-2 px-3 text-right font-mono">{value}</td>
    </tr>
  );
}
