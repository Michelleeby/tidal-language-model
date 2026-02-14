import { useMemo } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client.js";
import type { LogLine } from "@tidal/shared";

/**
 * Fetches historical logs via REST and merges with live SSE lines.
 * Deduplicates by timestamp to avoid showing the same line twice.
 */
export function useJobLogs(jobId: string | undefined) {
  const queryClient = useQueryClient();

  const { data: historicalData } = useQuery({
    queryKey: ["job-logs", jobId],
    queryFn: () => api.getJobLogs(jobId!),
    enabled: !!jobId,
    refetchInterval: 10000,
  });

  const liveLines =
    queryClient.getQueryData<LogLine[]>(["job-logs", jobId, "live"]) ?? [];
  const historicalLines = historicalData?.lines ?? [];

  const lines = useMemo(() => {
    if (historicalLines.length === 0 && liveLines.length === 0) return [];

    // Find the last timestamp in historical data
    const lastHistoricalTs =
      historicalLines.length > 0
        ? historicalLines[historicalLines.length - 1].timestamp
        : -1;

    // Append only live lines newer than historical data
    const newLive = liveLines.filter((l) => l.timestamp > lastHistoricalTs);
    return newLive.length > 0
      ? [...historicalLines, ...newLive]
      : historicalLines;
  }, [historicalLines, liveLines]);

  return {
    lines,
    totalLines: historicalData?.totalLines ?? 0,
  };
}
