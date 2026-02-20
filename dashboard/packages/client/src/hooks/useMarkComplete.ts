import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client.js";

/** Mark an experiment as completed via the dashboard API. */
export function useMarkComplete() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (expId: string) => api.markComplete(expId),
    onSuccess: (_data, expId) => {
      queryClient.invalidateQueries({ queryKey: ["status", expId] });
      queryClient.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
}
