import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client.js";

export function useModelFileTree() {
  return useQuery({
    queryKey: ["model-files"],
    queryFn: () => api.getModelFileTree(),
    staleTime: 5 * 60_000,
  });
}

export function useModelFile(filePath: string | null) {
  return useQuery({
    queryKey: ["model-file", filePath],
    queryFn: () => api.getModelFile(filePath!),
    enabled: !!filePath,
  });
}
