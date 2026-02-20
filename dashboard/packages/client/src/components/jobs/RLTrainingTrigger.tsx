import { useState } from "react";
import { useActiveJob, useCreateJob } from "../../hooks/useJobs.js";
import { useAllLMCheckpoints } from "../../hooks/useMetrics.js";
import { usePlugin } from "../../hooks/usePlugin.js";
import type { CheckpointInfo } from "@tidal/shared";

const RL_ELIGIBLE_PHASES = new Set(["foundational", "final"]);

/** Filter checkpoints to those eligible as base models for RL training. */
export function filterRLEligibleCheckpoints(
  checkpoints: CheckpointInfo[],
): CheckpointInfo[] {
  return checkpoints.filter((cp) => RL_ELIGIBLE_PHASES.has(cp.phase));
}

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
  const { data: allCheckpointsData } = useAllLMCheckpoints();
  const { data: activeData } = useActiveJob();
  const createJob = useCreateJob();
  const { manifest } = usePlugin();

  // Derive RL training phase config from manifest
  const rlPhase = manifest?.trainingPhases.find((p) => p.id === "rl-training");
  const pluginPrefix = manifest ? `plugins/${manifest.name}/` : "plugins/tidal/";
  const defaultConfigPath = rlPhase?.configFiles[0]
    ? `${pluginPrefix}${rlPhase.configFiles[0]}`
    : `${pluginPrefix}configs/base_config.yaml`;
  const defaultRlConfigPath = rlPhase?.configFiles[1]
    ? `${pluginPrefix}${rlPhase.configFiles[1]}`
    : `${pluginPrefix}configs/rl_config.yaml`;

  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
  const [configPath, setConfigPath] = useState(defaultConfigPath);
  const [rlConfigPath, setRlConfigPath] = useState(defaultRlConfigPath);
  const [timesteps, setTimesteps] = useState("");

  const activeJob = activeData?.job ?? null;
  const hasActiveRLJob =
    activeJob?.type === "rl-training" &&
    !["completed", "failed", "cancelled"].includes(activeJob.status);

  const groups = allCheckpointsData?.groups ?? [];

  // Default to current experiment's final checkpoint if available
  const currentExpGroup = groups.find((g) => g.experimentId === selectedExpId);
  const defaultCheckpoint = currentExpGroup?.checkpoints.at(-1)?.path ?? "";

  const handleStart = () => {
    const checkpoint = selectedCheckpoint || defaultCheckpoint;
    if (!checkpoint) return;
    createJob.mutate({
      type: "rl-training",
      plugin: manifest?.name ?? "tidal",
      configPath,
      rlConfigPath,
      checkpoint,
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
              value={selectedCheckpoint || defaultCheckpoint}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
              className="bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm text-gray-200 focus:outline-none focus:border-blue-500"
            >
              <option value="">Select model checkpoint</option>
              {groups.map((group) => (
                <optgroup
                  key={group.experimentId}
                  label={group.experimentId}
                >
                  {group.checkpoints.map((cp) => (
                    <option key={cp.path} value={cp.path}>
                      {cp.filename}
                      {cp.epoch != null ? ` (epoch ${cp.epoch})` : ""}
                    </option>
                  ))}
                </optgroup>
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
                (!selectedCheckpoint && !defaultCheckpoint) || createJob.isPending || hasActiveRLJob
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
