import type { TrainingStatus } from "@tidal/shared";

interface TrainingStatusCardProps {
  status: TrainingStatus | null;
}

export default function TrainingStatusCard({
  status,
}: TrainingStatusCardProps) {
  if (!status) {
    return (
      <div className="bg-gray-900 rounded-lg p-4 text-gray-500 text-sm">
        No status available
      </div>
    );
  }

  const statusColor = {
    initialized: "text-yellow-400",
    training: "text-blue-400",
    completed: "text-green-400",
  }[status.status];

  return (
    <div className="bg-gray-900 rounded-lg p-4 space-y-2">
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
    </div>
  );
}
