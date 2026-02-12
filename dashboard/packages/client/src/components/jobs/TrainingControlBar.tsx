import { useState } from "react";
import { useActiveJob, useCreateJob, useSignalJob, useCancelJob, useJobs } from "../../hooks/useJobs.js";
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

/** Statuses where the worker is actually running and can receive signals */
const SIGNALABLE: Set<JobStatus> = new Set(["running", "completing"]);

/** Statuses that are non-terminal but not signalable (stuck startup) */
const CANCELLABLE_STUCK: Set<JobStatus> = new Set([
  "pending",
  "provisioning",
  "starting",
  "stopping",
]);

export default function TrainingControlBar() {
  const { data: activeData } = useActiveJob();
  const { data: jobsData } = useJobs();
  const createJob = useCreateJob();
  const signalJob = useSignalJob();
  const cancelJob = useCancelJob();

  const [showStartDialog, setShowStartDialog] = useState(false);
  const [configPath, setConfigPath] = useState("configs/base_config.yaml");
  const [resumeDir, setResumeDir] = useState("");
  const [confirmStop, setConfirmStop] = useState(false);

  const activeJob = activeData?.job ?? null;
  const hasActiveJob = activeJob != null &&
    !["completed", "failed", "cancelled"].includes(activeJob.status);

  // Find most recent failed job (within last 5 minutes) to show error
  const recentFailedJob = !hasActiveJob
    ? (jobsData?.jobs ?? [])
        .filter((j) => j.status === "failed" && j.completedAt && Date.now() - j.completedAt < 300_000)
        .sort((a, b) => (b.completedAt ?? 0) - (a.completedAt ?? 0))[0] ?? null
    : null;

  const handleStart = () => {
    createJob.mutate(
      {
        type: "lm-training",
        configPath,
        resumeExpDir: resumeDir || undefined,
      },
      { onSuccess: () => setShowStartDialog(false) },
    );
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
    <div className="flex items-center gap-3 flex-wrap">
      {/* Status badge */}
      <div className="flex items-center gap-2 mr-2">
        <div
          className={`w-2.5 h-2.5 rounded-full ${
            activeJob ? STATUS_COLORS[activeJob.status] : "bg-gray-600"
          } ${activeJob?.status === "completing" ? "animate-pulse" : ""}`}
        />
        <span className="text-sm text-gray-400">
          {activeJob ? STATUS_LABELS[activeJob.status] : "No active job"}
        </span>
      </div>

      {/* Start Training */}
      {!showStartDialog ? (
        <button
          onClick={() => setShowStartDialog(true)}
          disabled={hasActiveJob || createJob.isPending}
          className="px-3 py-1.5 text-sm rounded bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Start Training
        </button>
      ) : (
        <div className="flex items-center gap-2 bg-gray-800 border border-gray-700 rounded px-3 py-2">
          <input
            type="text"
            value={configPath}
            onChange={(e) => setConfigPath(e.target.value)}
            placeholder="Config path"
            className="bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm text-gray-200 w-52"
          />
          <input
            type="text"
            value={resumeDir}
            onChange={(e) => setResumeDir(e.target.value)}
            placeholder="Resume dir (optional)"
            className="bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm text-gray-200 w-52"
          />
          <button
            onClick={handleStart}
            disabled={createJob.isPending || !configPath}
            className="px-3 py-1 text-sm rounded bg-green-600 hover:bg-green-500 text-white disabled:opacity-40 transition-colors"
          >
            {createJob.isPending ? "Starting..." : "Go"}
          </button>
          <button
            onClick={() => setShowStartDialog(false)}
            className="px-2 py-1 text-sm text-gray-400 hover:text-gray-200"
          >
            Cancel
          </button>
        </div>
      )}

      {/* Complete After Epoch — only when actually running */}
      {activeJob?.status === "running" && (
        <button
          onClick={handleComplete}
          disabled={signalJob.isPending}
          className="px-3 py-1.5 text-sm rounded bg-amber-600 hover:bg-amber-500 text-white disabled:opacity-40 transition-colors"
        >
          Complete After Epoch
        </button>
      )}

      {/* Stop Now — for running/completing jobs (sends signal + kill) */}
      {activeJob && SIGNALABLE.has(activeJob.status) && (
        <button
          onClick={handleStop}
          disabled={signalJob.isPending}
          className={`px-3 py-1.5 text-sm rounded text-white transition-colors ${
            confirmStop
              ? "bg-red-700 hover:bg-red-600"
              : "bg-red-600/70 hover:bg-red-600"
          } disabled:opacity-40`}
        >
          {confirmStop ? "Confirm Stop" : "Stop Now"}
        </button>
      )}

      {/* Cancel — for stuck jobs (pending/provisioning/starting/stopping) */}
      {activeJob && CANCELLABLE_STUCK.has(activeJob.status) && (
        <button
          onClick={handleCancel}
          disabled={cancelJob.isPending}
          className="px-3 py-1.5 text-sm rounded bg-red-600/70 hover:bg-red-600 text-white disabled:opacity-40 transition-colors"
        >
          {cancelJob.isPending ? "Cancelling..." : "Cancel Job"}
        </button>
      )}

      {/* Error display */}
      {createJob.isError && (
        <span className="text-sm text-red-400">
          {(createJob.error as Error).message}
        </span>
      )}
      {recentFailedJob && (
        <span className="text-sm text-red-400">
          Last job failed: {recentFailedJob.error ?? "Unknown error"}
        </span>
      )}
    </div>
  );
}
