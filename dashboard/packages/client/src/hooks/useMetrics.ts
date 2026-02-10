import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client.js";

export function useMetrics(
  expId: string | null,
  mode: "recent" | "historical" = "recent",
  window = 5000,
  maxPoints = 2000,
) {
  return useQuery({
    queryKey: ["metrics", expId, mode, window, maxPoints],
    queryFn: () => api.getMetrics(expId!, mode, window, maxPoints),
    enabled: !!expId,
  });
}

export function useRLMetrics(expId: string | null) {
  return useQuery({
    queryKey: ["rl-metrics", expId],
    queryFn: () => api.getRLMetrics(expId!),
    enabled: !!expId,
  });
}

export function useCheckpoints(expId: string | null) {
  return useQuery({
    queryKey: ["checkpoints", expId],
    queryFn: () => api.getCheckpoints(expId!),
    enabled: !!expId,
  });
}

export function useEvaluation(expId: string | null) {
  return useQuery({
    queryKey: ["evaluation", expId],
    queryFn: () => api.getEvaluation(expId!),
    enabled: !!expId,
  });
}

export function useAblation(expId: string | null) {
  return useQuery({
    queryKey: ["ablation", expId],
    queryFn: () => api.getAblation(expId!),
    enabled: !!expId,
  });
}
