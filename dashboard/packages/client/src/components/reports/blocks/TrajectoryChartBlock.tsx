import { useState } from "react";
import { createReactBlockSpec } from "@blocknote/react";
import { defaultProps } from "@blocknote/core";
import { useMutation } from "@tanstack/react-query";
import { api } from "../../../api/client.js";
import { useExperiments } from "../../../hooks/useExperiments.js";
import { useCheckpoints } from "../../../hooks/useMetrics.js";
import { useAnalyses, useAnalysis, useSaveAnalysis } from "../../../hooks/useAnalyses.js";
import GateTrajectoryChart from "../../charts/GateTrajectoryChart.js";
import type { GenerateRequest, GenerateResponse } from "@tidal/shared";

/** Return gating mode options for the selector. */
export function gatingModeOptions(): Array<{ value: string; label: string }> {
  return [
    { value: "fixed", label: "Fixed" },
    { value: "random", label: "Random" },
    { value: "learned", label: "Learned (RL)" },
  ];
}

export const TrajectoryChartBlock = createReactBlockSpec(
  {
    type: "trajectoryChart" as const,
    propSchema: {
      ...defaultProps,
      experimentId: { default: "" },
      gatingMode: { default: "fixed" },
      prompt: { default: "Once upon a time," },
      analysisId: { default: "" },
    },
    content: "none",
  },
  {
    render: ({ block, editor }: any) => {
      const { experimentId, gatingMode, prompt, analysisId } = block.props;
      const { data: expData } = useExperiments();
      const { data: checkpointsData } = useCheckpoints(experimentId || null);
      const { data: cachedAnalyses } = useAnalyses(experimentId || null, "trajectory");
      const { data: cachedData } = useAnalysis(analysisId || null);
      const saveAnalysis = useSaveAnalysis();
      const isEditable = editor.isEditable;

      const experiments = expData?.experiments ?? [];
      const checkpoints = checkpointsData?.checkpoints ?? [];

      // Auto-resolve first foundational checkpoint
      const checkpoint = checkpoints.find((cp) => cp.phase === "foundational")?.path
        ?? checkpoints[0]?.path
        ?? "";

      const [result, setResult] = useState<GenerateResponse | null>(null);

      // Use cached data if available and no live result yet
      const cachedResult = cachedData?.analysis?.data as unknown as GenerateResponse | undefined;
      const displayResult = result ?? cachedResult ?? null;

      const generateMutation = useMutation({
        mutationFn: (body: GenerateRequest) => api.generate(body),
        onSuccess: (data) => {
          setResult(data);
          // Auto-save to cache and set analysisId
          if (data.trajectory && experimentId) {
            saveAnalysis.mutate(
              {
                expId: experimentId,
                analysisType: "trajectory",
                label: `Trajectory — ${gatingMode} — "${prompt.slice(0, 30)}"`,
                request: { checkpoint, prompt, gatingMode } as Record<string, unknown>,
                data: data as unknown as Record<string, unknown>,
              },
              {
                onSuccess: (resp) => {
                  editor.updateBlock(block, {
                    props: { analysisId: resp.analysis.id },
                  });
                },
              },
            );
          }
        },
      });

      const handleGenerate = () => {
        if (!checkpoint) return;
        generateMutation.mutate({
          checkpoint,
          prompt,
          maxTokens: 50,
          temperature: 0.8,
          topK: 50,
          gatingMode: gatingMode as GenerateRequest["gatingMode"],
        });
      };

      const analyses = cachedAnalyses?.analyses ?? [];
      const modes = gatingModeOptions();

      return (
        <div className="my-2 rounded-lg border border-gray-700 bg-gray-900 p-4">
          {isEditable && (
            <div className="mb-3 space-y-2">
              <div className="flex flex-wrap items-center gap-2">
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
                <select
                  className="rounded bg-gray-800 border border-gray-600 text-gray-200 text-sm px-2 py-1"
                  value={gatingMode}
                  onChange={(e) => {
                    editor.updateBlock(block, {
                      props: { gatingMode: e.target.value },
                    });
                  }}
                >
                  {modes.map((m) => (
                    <option key={m.value} value={m.value}>
                      {m.label}
                    </option>
                  ))}
                </select>
                <button
                  onClick={handleGenerate}
                  disabled={!checkpoint || generateMutation.isPending}
                  className="px-3 py-1 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {generateMutation.isPending ? "Generating..." : "Generate"}
                </button>
              </div>
              <textarea
                className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200 resize-none"
                rows={2}
                value={prompt}
                onChange={(e) => {
                  editor.updateBlock(block, {
                    props: { prompt: e.target.value },
                  });
                }}
                placeholder="Enter prompt..."
              />
              {analyses.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500">Cached:</span>
                  <select
                    className="rounded bg-gray-800 border border-gray-600 text-gray-200 text-xs px-2 py-1 flex-1"
                    value={analysisId}
                    onChange={(e) => {
                      editor.updateBlock(block, {
                        props: { analysisId: e.target.value },
                      });
                    }}
                  >
                    <option value="">Select cached analysis...</option>
                    {analyses.map((a) => (
                      <option key={a.id} value={a.id}>
                        {a.label} ({new Date(a.createdAt).toLocaleString()})
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          )}

          {!experimentId ? (
            <div className="text-gray-500 text-sm py-8 text-center">
              Select an experiment to generate a trajectory
            </div>
          ) : generateMutation.isPending ? (
            <div className="text-gray-500 text-sm py-8 text-center">
              Generating trajectory...
            </div>
          ) : displayResult?.trajectory ? (
            <GateTrajectoryChart trajectory={displayResult.trajectory} />
          ) : (
            <div className="text-gray-500 text-sm py-8 text-center">
              Click &ldquo;Generate&rdquo; to create a trajectory
            </div>
          )}

          {generateMutation.isError && (
            <div className="bg-red-900/30 border border-red-800 rounded-lg p-3 text-sm text-red-300 mt-2">
              {(generateMutation.error as Error).message}
            </div>
          )}
        </div>
      );
    },
  },
);
