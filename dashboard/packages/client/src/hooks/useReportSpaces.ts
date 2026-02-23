import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client.js";
import type { SaveReportRequest } from "@tidal/shared";

/** Explicitly save a report to DigitalOcean Spaces (creates a versioned snapshot). */
export function useSaveReport() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...body }: SaveReportRequest & { id: string }) =>
      api.saveReport(id, body),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ["reports", variables.id] });
      queryClient.invalidateQueries({ queryKey: ["reports"] });
      queryClient.invalidateQueries({ queryKey: ["reportVersions", variables.id] });
    },
  });
}

/** List saved Spaces versions for a report. */
export function useReportVersions(id: string | null) {
  return useQuery({
    queryKey: ["reportVersions", id],
    queryFn: () => api.getReportVersions(id!),
    enabled: !!id,
  });
}

/** Restore a historical Spaces version of a report. */
export function useRestoreReportVersion() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, timestamp }: { id: string; timestamp: number }) =>
      api.restoreReportVersion(id, timestamp),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ["reports", variables.id] });
      queryClient.invalidateQueries({ queryKey: ["reports"] });
      queryClient.invalidateQueries({ queryKey: ["reportVersions", variables.id] });
    },
  });
}
