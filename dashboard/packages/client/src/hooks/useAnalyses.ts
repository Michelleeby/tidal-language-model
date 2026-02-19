import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client.js";
import type { CreateAnalysisRequest, AnalysisType } from "@tidal/shared";

/** List cached analysis summaries for an experiment, optionally filtered by type. */
export function useAnalyses(expId: string | null, type?: AnalysisType) {
  return useQuery({
    queryKey: ["analyses", expId, type],
    queryFn: () => api.listAnalyses(expId!, type),
    enabled: !!expId,
  });
}

/** Fetch a full analysis result (with data blob) by ID. */
export function useAnalysis(id: string | null) {
  return useQuery({
    queryKey: ["analysis", id],
    queryFn: () => api.getAnalysis(id!),
    enabled: !!id,
  });
}

/** Save an analysis result to the server cache. */
export function useSaveAnalysis() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      expId,
      ...body
    }: CreateAnalysisRequest & { expId: string }) =>
      api.createAnalysis(expId, body),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["analyses", variables.expId],
      });
    },
  });
}

/** Delete a cached analysis result. */
export function useDeleteAnalysis() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteAnalysis(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["analyses"] });
    },
  });
}
