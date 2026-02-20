import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client.js";
import type { CreateJobRequest, CreateJobResponse, JobSignal } from "@tidal/shared";
import { useExperimentStore } from "../stores/experimentStore.js";

export function useJobs() {
  return useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.getJobs(),
    refetchInterval: 5000,
  });
}

export function useActiveJob() {
  return useQuery({
    queryKey: ["jobs", "active"],
    queryFn: () => api.getActiveJob(),
    refetchInterval: 3000,
  });
}

export function useCreateJob() {
  const queryClient = useQueryClient();
  const setSelectedExpId = useExperimentStore((s) => s.setSelectedExpId);
  return useMutation({
    mutationFn: (body: CreateJobRequest) => api.createJob(body),
    onSuccess: (data: CreateJobResponse) => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      if (data.job.experimentId) {
        queryClient.invalidateQueries({ queryKey: ["experiments"] });
        setSelectedExpId(data.job.experimentId);
      }
    },
  });
}

export function useSignalJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ jobId, signal }: { jobId: string; signal: JobSignal }) =>
      api.signalJob(jobId, signal),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

export function useCancelJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api.cancelJob(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}
