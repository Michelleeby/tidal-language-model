import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { MetricPoint, TrainingStatus, RLTrainingMetrics } from "@tidal/shared";

/**
 * SSE hook that connects to the experiment's event stream and
 * pushes updates into React Query cache for automatic re-renders.
 */
export function useSSE(expId: string | null) {
  const queryClient = useQueryClient();
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!expId) return;

    const source = new EventSource(`/api/experiments/${expId}/stream`);
    sourceRef.current = source;

    source.addEventListener("metrics", (e) => {
      const point: MetricPoint = JSON.parse(e.data);
      queryClient.setQueryData(
        ["metrics", expId, "live"],
        (old: MetricPoint[] | undefined) => {
          const prev = old ?? [];
          // Keep last 5000 points in the live cache
          const next = [...prev, point];
          return next.length > 5000 ? next.slice(-5000) : next;
        },
      );
    });

    source.addEventListener("status", (e) => {
      const status: TrainingStatus = JSON.parse(e.data);
      queryClient.setQueryData(["status", expId], { expId, status });
    });

    source.addEventListener("rl-metrics", (e) => {
      const metrics: RLTrainingMetrics = JSON.parse(e.data);
      queryClient.setQueryData(["rl-metrics", expId], { expId, metrics });
    });

    return () => {
      source.close();
      sourceRef.current = null;
    };
  }, [expId, queryClient]);
}
