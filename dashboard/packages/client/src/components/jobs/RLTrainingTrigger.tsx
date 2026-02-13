import { useState } from "react";
import { useActiveJob, useCreateJob } from "../../hooks/useJobs.js";
import { useCheckpoints } from "../../hooks/useMetrics.js";
import type { JobStatus } from "@tidal/shared";

const STATUS_COLORS: Record<string, string> = {
  running: "text-green-400",
  completing: "text-amber-400",
  stopping: "text-red-400",
  failed: "text-red-500",
};

interface Props {
  selectedExpId: string | null;
}

export default function RLTrainingTrigger({ selectedExpId }: Props) {
  const { data: checkpointsData } = useCheckpoints(selectedExpId);
  const { data: activeData } = useActiveJob();
  const createJob = useCreateJob();
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
  const [configPath, setConfigPath] = useState("configs/base_config.yaml");
  const [rlConfigPath, setRlConfigPath] = useState("configs/rl_config.yaml");
  const [timesteps, setTimesteps] = useState("");

  const activeJob = activeData?.job ?? null;
  const hasActiveRLJob =
    activeJob?.type === "rl-training" &&
    !["completed", "failed", "cancelled"].includes(activeJob.status);

  // Filter to foundational phase checkpoints only
  const foundationalCheckpoints = (checkpointsData?.checkpoints ?? []).filter(
    (cp) => cp.phase === "foundational",
  );

  const handleStart = () => {
    if (!selectedCheckpoint) return;
    createJob.mutate({
      type: "rl-training",
      configPath,
      rlConfigPath,
      checkpoint: selectedCheckpoint,
      timesteps: timesteps ? parseInt(timesteps, 10) : undefined,
    });
  };

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-lg px-4 py-3">
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-sm font-medium text-gray-300">
          RL Training:
        </span>

        {hasActiveRLJob ? (
          <span
            className={`text-sm ${STATUS_COLORS[activeJob!.status] ?? "text-gray-400"}`}
          >
            {activeJob!.status === "running"
              ? "RL training in progress..."
              : `Status: ${activeJob!.status}`}
          </span>
        ) : (
          <>
            <select
              value={selectedCheckpoint}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
              className="bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm text-gray-200 focus:outline-none focus:border-blue-500"
            >
              <option value="">Select epoch checkpoint</option>
              {foundationalCheckpoints.map((cp) => (
                <option key={cp.path} value={cp.path}>
                  {cp.filename}
                  {cp.epoch != null ? ` (epoch ${cp.epoch})` : ""}
                </option>
              ))}
            </select>

            <input
              type="text"
              value={configPath}
              onChange={(e) => setConfigPath(e.target.value)}
              placeholder="Base config"
              className="bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm text-gray-200 w-44"
            />
            <input
              type="text"
              value={rlConfigPath}
              onChange={(e) => setRlConfigPath(e.target.value)}
              placeholder="RL config"
              className="bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm text-gray-200 w-40"
            />
            <input
              type="text"
              value={timesteps}
              onChange={(e) => setTimesteps(e.target.value.replace(/\D/g, ""))}
              placeholder="Timesteps (optional)"
              className="bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm text-gray-200 w-36"
            />

            <button
              onClick={handleStart}
              disabled={
                !selectedCheckpoint || createJob.isPending || hasActiveRLJob
              }
              className="px-3 py-1.5 text-sm rounded bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {createJob.isPending ? "Starting..." : "Start RL Training"}
            </button>
          </>
        )}

        {createJob.isError && (
          <span className="text-sm text-red-400">
            {(createJob.error as Error).message}
          </span>
        )}
      </div>
    </div>
  );
}
