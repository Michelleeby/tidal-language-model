import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client.js";
import type { UpdateReportRequest, CreateReportRequest } from "@tidal/shared";

export function useReports() {
  return useQuery({
    queryKey: ["reports"],
    queryFn: () => api.getReports(),
  });
}

export function useReport(id: string | null) {
  return useQuery({
    queryKey: ["reports", id],
    queryFn: () => api.getReport(id!),
    enabled: !!id,
  });
}

export function useCreateReport() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateReportRequest | void) => api.createReport(body ?? undefined),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["reports"] });
    },
  });
}

export function useUpdateReport() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...body }: UpdateReportRequest & { id: string }) =>
      api.updateReport(id, body),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ["reports", variables.id] });
      queryClient.invalidateQueries({ queryKey: ["reports"] });
    },
  });
}

export function useDeleteReport() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteReport(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["reports"] });
    },
  });
}
