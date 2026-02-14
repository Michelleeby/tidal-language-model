import { useJobLogs } from "../../hooks/useLogs.js";

const TAIL_LINES = 5;

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-US", { hour12: false });
}

interface LogTailCardProps {
  jobId: string | undefined;
}

export default function LogTailCard({ jobId }: LogTailCardProps) {
  const { lines } = useJobLogs(jobId);

  const tail = lines.slice(-TAIL_LINES);

  if (!jobId) {
    return (
      <div className="bg-gray-950 border border-gray-800 rounded-lg p-4">
        <div className="text-gray-600 text-xs font-mono text-center">
          No active job
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-950 border border-gray-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-gray-800">
        <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">
          Output
        </span>
        <span className="text-[10px] text-gray-600">
          {lines.length > 0
            ? `${lines.length.toLocaleString()} lines`
            : "Waiting..."}
        </span>
      </div>

      {/* Log content */}
      <div className="font-mono text-[11px] leading-[18px] p-2">
        {tail.length === 0 ? (
          <div className="text-gray-600 text-center py-4">
            Waiting for output...
          </div>
        ) : (
          tail.map((line, i) => (
            <div key={i} className="flex gap-2 hover:bg-gray-900/50 px-1">
              <span className="text-gray-700 select-none shrink-0">
                {formatTime(line.timestamp)}
              </span>
              <span
                className={`whitespace-pre-wrap break-all ${
                  line.stream === "stderr"
                    ? "text-red-400"
                    : "text-gray-400"
                }`}
              >
                {line.line}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
