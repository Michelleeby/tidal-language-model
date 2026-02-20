import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { TrainingStatus } from "@tidal/shared";
import { api } from "../../api/client.js";

interface TrainingStatusCardProps {
  status: TrainingStatus | null;
  expId?: string;
}

export default function TrainingStatusCard({
  status,
  expId,
}: TrainingStatusCardProps) {
  const queryClient = useQueryClient();
  const [marking, setMarking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!status) {
    return (
      <div className="snap-center flex-shrink-0 w-full md:w-auto bg-gray-900 rounded-lg p-4 text-gray-500 text-sm">
        No status available
      </div>
    );
  }

  const statusColor = {
    initialized: "text-yellow-400",
    training: "text-blue-400",
    completed: "text-green-400",
  }[status.status];

  const handleMarkComplete = async () => {
    if (!expId || marking) return;
    setMarking(true);
    setError(null);
    try {
      await api.markComplete(expId);
      await queryClient.invalidateQueries({ queryKey: ["status", expId] });
      await queryClient.invalidateQueries({ queryKey: ["experiments"] });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to mark complete");
    } finally {
      setMarking(false);
    }
  };

  return (
    <div className="snap-center flex-shrink-0 w-full md:w-auto bg-gray-900 rounded-lg p-4 space-y-2">
      <div className="flex items-center gap-2">
        <span
          className={`inline-block w-2 h-2 rounded-full ${
            status.status === "training" ? "bg-blue-400 animate-pulse" : "bg-green-400"
          }`}
        />
        <span className={`text-sm font-medium ${statusColor}`}>
          {status.status.charAt(0).toUpperCase() + status.status.slice(1)}
        </span>
      </div>
      {status.current_step !== undefined && (
        <div className="text-xs text-gray-400">
          Step: {status.current_step.toLocaleString()}
        </div>
      )}
      {status.total_metrics_logged !== undefined && (
        <div className="text-xs text-gray-400">
          Points logged: {status.total_metrics_logged.toLocaleString()}
        </div>
      )}
      {expId && status.status !== "completed" && (
        <button
          type="button"
          onClick={handleMarkComplete}
          disabled={marking}
          className="mt-1 text-xs text-gray-500 hover:text-gray-300 border border-gray-700 hover:border-gray-500 rounded px-2 py-0.5 transition-colors disabled:opacity-50"
        >
          {marking ? "Marking..." : "Mark Complete"}
        </button>
      )}
      {error && (
        <div className="text-xs text-red-400 mt-1">{error}</div>
      )}
    </div>
  );
}
