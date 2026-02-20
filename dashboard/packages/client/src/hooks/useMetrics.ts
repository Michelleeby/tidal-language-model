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

export function useFullMetrics(expId: string | null, maxPoints = 5000) {
  return useQuery({
    queryKey: ["metrics", expId, "historical", 0, maxPoints],
    queryFn: () => api.getMetrics(expId!, "historical", 0, maxPoints),
    enabled: !!expId,
    staleTime: 60_000,
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

export function useAllLMCheckpoints() {
  return useQuery({
    queryKey: ["all-checkpoints"],
    queryFn: () => api.getAllCheckpoints(),
    staleTime: 30_000,
  });
}
