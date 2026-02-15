import { useState } from "react";
import { useUserPlugins } from "../../hooks/useUserPlugins.js";
import { useUserPluginManifest } from "../../hooks/useUserPluginManifest.js";
import { usePluginGitStatus } from "../../hooks/usePluginGit.js";
import {
  useActiveJob,
  useCreateJob,
  useSignalJob,
  useCancelJob,
  useJobs,
} from "../../hooks/useJobs.js";
import type { JobStatus } from "@tidal/shared";

const STATUS_COLORS: Record<JobStatus, string> = {
  pending: "bg-yellow-500",
  provisioning: "bg-yellow-500",
  starting: "bg-yellow-500",
  running: "bg-green-500",
  completing: "bg-amber-500",
  stopping: "bg-red-400",
  completed: "bg-gray-500",
  failed: "bg-red-600",
  cancelled: "bg-gray-500",
};

const STATUS_LABELS: Record<JobStatus, string> = {
  pending: "Pending",
  provisioning: "Provisioning",
  starting: "Starting",
  running: "Running",
  completing: "Completing Epoch...",
  stopping: "Stopping...",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

const SIGNALABLE: Set<JobStatus> = new Set(["running", "completing"]);
const CANCELLABLE_STUCK: Set<JobStatus> = new Set([
  "pending",
  "provisioning",
  "starting",
  "stopping",
]);

interface TrainingPhase {
  id: string;
  name: string;
  configFiles?: string[];
}

/**
 * Allows users to start training jobs for their user plugins.
 * Renders inside the experiments page alongside TrainingControlBar.
 * Shows a plugin selector + phase buttons + git dirty warning.
 */
export default function PluginTrainingPanel() {
  const { data: pluginsData } = useUserPlugins();
  const plugins = pluginsData?.plugins ?? [];

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [confirmStop, setConfirmStop] = useState(false);

  const { data: manifestData } = useUserPluginManifest(selectedId);
  const { data: gitStatus } = usePluginGitStatus(selectedId);
  const { data: activeData } = useActiveJob();
  const { data: jobsData } = useJobs();
  const createJob = useCreateJob();
  const signalJob = useSignalJob();
  const cancelJob = useCancelJob();

  // No user plugins — don't render anything
  if (plugins.length === 0) return null;

  const selectedPlugin = plugins.find((p) => p.id === selectedId);
  const manifest = manifestData?.manifest as
    | { name?: string; trainingPhases?: TrainingPhase[] }
    | undefined;
  const phases = manifest?.trainingPhases ?? [];
  const pluginName = manifest?.name ?? selectedPlugin?.name ?? "";
  const dirty = gitStatus?.dirty ?? false;

  const activeJob = activeData?.job ?? null;
  const allJobs = jobsData?.jobs ?? [];
  const hasActiveJob = allJobs.some(
    (j) => !["completed", "failed", "cancelled"].includes(j.status),
  );

  const handleStartPhase = (phase: TrainingPhase) => {
    if (dirty || !selectedId) return;
    const configPath = phase.configFiles?.[0]
      ? `user-plugins/${selectedId}/${phase.configFiles[0]}`
      : "";
    createJob.mutate({
      type: phase.id,
      plugin: pluginName,
      configPath,
      userPluginId: selectedId,
    });
  };

  const handleComplete = () => {
    if (!activeJob) return;
    signalJob.mutate({ jobId: activeJob.jobId, signal: "complete" });
  };

  const handleStop = () => {
    if (!activeJob) return;
    if (!confirmStop) {
      setConfirmStop(true);
      setTimeout(() => setConfirmStop(false), 3000);
      return;
    }
    signalJob.mutate({ jobId: activeJob.jobId, signal: "stop" });
    setConfirmStop(false);
  };

  const handleCancel = () => {
    if (!activeJob) return;
    cancelJob.mutate(activeJob.jobId);
  };

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-3">
      <h3 className="text-sm font-semibold text-gray-300">
        User Plugin Training
      </h3>

      {/* Plugin selector */}
      <select
        value={selectedId ?? ""}
        onChange={(e) => setSelectedId(e.target.value || null)}
        className="w-full px-2 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded text-gray-200 outline-none focus:border-blue-500"
      >
        <option value="">Select a plugin...</option>
        {plugins.map((p) => (
          <option key={p.id} value={p.id}>
            {p.displayName} ({p.name})
          </option>
        ))}
      </select>

      {selectedId && (
        <>
          {/* Active job status */}
          {activeJob && (
            <div className="flex items-center gap-2 flex-wrap">
              <div
                className={`w-2.5 h-2.5 rounded-full ${STATUS_COLORS[activeJob.status]} ${
                  activeJob.status === "completing" ? "animate-pulse" : ""
                }`}
              />
              <span className="text-sm text-gray-400">
                {STATUS_LABELS[activeJob.status]}
              </span>

              {activeJob.status === "running" && (
                <button
                  onClick={handleComplete}
                  disabled={signalJob.isPending}
                  className="px-3 py-1 text-xs rounded bg-amber-600 hover:bg-amber-500 text-white disabled:opacity-40 transition-colors"
                >
                  Complete After Epoch
                </button>
              )}
              {SIGNALABLE.has(activeJob.status) && (
                <button
                  onClick={handleStop}
                  disabled={signalJob.isPending}
                  className={`px-3 py-1 text-xs rounded text-white transition-colors ${
                    confirmStop
                      ? "bg-red-700 hover:bg-red-600"
                      : "bg-red-600/70 hover:bg-red-600"
                  } disabled:opacity-40`}
                >
                  {confirmStop ? "Confirm Stop" : "Stop Now"}
                </button>
              )}
              {CANCELLABLE_STUCK.has(activeJob.status) && (
                <button
                  onClick={handleCancel}
                  disabled={cancelJob.isPending}
                  className="px-3 py-1 text-xs rounded bg-red-600/70 hover:bg-red-600 text-white disabled:opacity-40 transition-colors"
                >
                  {cancelJob.isPending ? "Cancelling..." : "Cancel Job"}
                </button>
              )}
            </div>
          )}

          {/* Dirty warning */}
          {dirty && (
            <p className="text-xs text-yellow-400">
              Push changes before starting training
            </p>
          )}

          {/* Training phases */}
          {phases.length === 0 ? (
            <p className="text-xs text-gray-500">
              {manifestData
                ? "No training phases found in manifest"
                : "Loading manifest..."}
            </p>
          ) : (
            <div className="flex gap-2 flex-wrap">
              {phases.map((phase) => (
                <button
                  key={phase.id}
                  onClick={() => handleStartPhase(phase)}
                  disabled={hasActiveJob || createJob.isPending || dirty}
                  className="px-3 py-1.5 text-sm rounded bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  {createJob.isPending
                    ? "Starting..."
                    : `Start ${phase.name ?? phase.id}`}
                </button>
              ))}
            </div>
          )}

          {/* Error display */}
          {createJob.isError && (
            <p className="text-sm text-red-400">
              {(createJob.error as Error).message}
            </p>
          )}
        </>
      )}
    </div>
  );
}
