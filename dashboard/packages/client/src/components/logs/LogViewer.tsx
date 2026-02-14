import { useEffect, useRef, useState } from "react";
import { useJobLogs } from "../../hooks/useLogs.js";

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-US", { hour12: false });
}

interface LogViewerProps {
  jobId: string | undefined;
}

export default function LogViewer({ jobId }: LogViewerProps) {
  const { lines, totalLines } = useJobLogs(jobId);
  const containerRef = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);

  // Auto-scroll when follow is enabled and new lines arrive
  useEffect(() => {
    if (follow && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [lines.length, follow]);

  // Detect manual scroll to disable follow
  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const atBottom = scrollHeight - scrollTop - clientHeight < 40;
    if (follow && !atBottom) {
      setFollow(false);
    } else if (!follow && atBottom) {
      setFollow(true);
    }
  };

  if (!jobId) {
    return (
      <div className="text-gray-500 text-sm py-8 text-center">
        No active job. Start a training job to see logs here.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-400">
          {totalLines > 0 ? `${totalLines.toLocaleString()} lines` : "Waiting for output..."}
        </span>
        <button
          onClick={() => {
            setFollow(!follow);
            if (!follow && containerRef.current) {
              containerRef.current.scrollTop = containerRef.current.scrollHeight;
            }
          }}
          className={`px-3 py-1 text-xs rounded border transition-colors ${
            follow
              ? "border-blue-500 text-blue-400 bg-blue-500/10"
              : "border-gray-700 text-gray-500 hover:text-gray-300"
          }`}
        >
          {follow ? "Following" : "Follow"}
        </button>
      </div>

      {/* Log container */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="bg-gray-950 border border-gray-800 rounded-lg font-mono text-xs leading-5 overflow-auto"
        style={{ maxHeight: "70vh", minHeight: "300px" }}
      >
        {lines.length === 0 ? (
          <div className="text-gray-600 p-4 text-center">
            No output yet...
          </div>
        ) : (
          <table className="w-full">
            <tbody>
              {lines.map((line, i) => (
                <tr
                  key={i}
                  className="hover:bg-gray-900/50"
                >
                  <td className="text-gray-600 select-none px-3 py-0 whitespace-nowrap align-top text-right w-0">
                    {formatTime(line.timestamp)}
                  </td>
                  <td
                    className={`px-2 py-0 whitespace-pre-wrap break-all ${
                      line.stream === "stderr"
                        ? "text-red-400"
                        : "text-gray-300"
                    }`}
                  >
                    {line.line}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
