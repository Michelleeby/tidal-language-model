import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client.js";

/** Delete an experiment (disk + Redis + analyses). */
export function useDeleteExperiment() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (expId: string) => api.deleteExperiment(expId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
}

/** Archive an experiment to DigitalOcean Spaces. */
export function useArchiveExperiment() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (expId: string) => api.archiveExperiment(expId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
}
