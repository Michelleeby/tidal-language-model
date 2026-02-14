import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { TrainingJob, LogLine } from "@tidal/shared";
import { useExperimentStore } from "../stores/experimentStore.js";

/**
 * SSE hook that connects to the global job event stream and
 * pushes job updates into React Query cache for instant re-renders.
 * Also auto-navigates to new experiments when a job reports its experimentId.
 */
export function useJobSSE() {
  const queryClient = useQueryClient();
  const sourceRef = useRef<EventSource | null>(null);
  const setSelectedExpId = useExperimentStore((s) => s.setSelectedExpId);
  // Track which job IDs we've already navigated for (avoid re-navigating)
  const navigatedJobsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const source = new EventSource("/api/jobs/stream");
    sourceRef.current = source;

    source.addEventListener("job-update", (e) => {
      const job: TrainingJob = JSON.parse(e.data);

      // Auto-navigate to newly created experiment
      if (
        job.experimentId &&
        !navigatedJobsRef.current.has(job.jobId)
      ) {
        navigatedJobsRef.current.add(job.jobId);
        // Refresh experiment list so the new experiment appears in sidebar
        queryClient.invalidateQueries({ queryKey: ["experiments"] });
        setSelectedExpId(job.experimentId);
      }

      // Update the active job query
      queryClient.setQueryData(["jobs", "active"], { job });

      // Update the jobs list cache
      queryClient.invalidateQueries({ queryKey: ["jobs"] });

      // Update individual job cache
      queryClient.setQueryData(["jobs", job.jobId], { job });
    });

    source.addEventListener("log-lines", (e) => {
      const { jobId, lines } = JSON.parse(e.data) as {
        jobId: string;
        lines: LogLine[];
      };

      queryClient.setQueryData(
        ["job-logs", jobId, "live"],
        (old: LogLine[] | undefined) => {
          const prev = old ?? [];
          const next = [...prev, ...lines];
          // Keep last 10000 lines in live cache
          return next.length > 10000 ? next.slice(-10000) : next;
        },
      );
    });

    return () => {
      source.close();
      sourceRef.current = null;
    };
  }, [queryClient]);
}
