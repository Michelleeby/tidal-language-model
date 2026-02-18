import { useMutation } from "@tanstack/react-query";
import { api } from "../api/client.js";
import type { AnalyzeRequest } from "@tidal/shared";

export function useTrajectoryAnalysis() {
  return useMutation({
    mutationFn: (body: AnalyzeRequest) => api.analyzeTrajectories(body),
  });
}
