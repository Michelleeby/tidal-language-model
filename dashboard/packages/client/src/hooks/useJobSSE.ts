import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { TrainingJob } from "@tidal/shared";

/**
 * SSE hook that connects to the global job event stream and
 * pushes job updates into React Query cache for instant re-renders.
 */
export function useJobSSE() {
  const queryClient = useQueryClient();
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const source = new EventSource("/api/jobs/stream");
    sourceRef.current = source;

    source.addEventListener("job-update", (e) => {
      const job: TrainingJob = JSON.parse(e.data);

      // Update the active job query
      queryClient.setQueryData(["jobs", "active"], { job });

      // Update the jobs list cache
      queryClient.invalidateQueries({ queryKey: ["jobs"] });

      // Update individual job cache
      queryClient.setQueryData(["jobs", job.jobId], { job });
    });

    return () => {
      source.close();
      sourceRef.current = null;
    };
  }, [queryClient]);
}
