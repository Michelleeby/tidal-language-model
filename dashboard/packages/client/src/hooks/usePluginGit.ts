import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client.js";

export function usePluginGitStatus(pluginId: string | null) {
  return useQuery({
    queryKey: ["user-plugins", pluginId, "git-status"],
    queryFn: () => api.getPluginGitStatus(pluginId!),
    enabled: !!pluginId,
    refetchInterval: 10_000,
  });
}

export function usePullPlugin() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (pluginId: string) => api.pullPlugin(pluginId),
    onSuccess: (_data, pluginId) => {
      queryClient.invalidateQueries({
        queryKey: ["user-plugins", pluginId, "git-status"],
      });
      queryClient.invalidateQueries({
        queryKey: ["plugin-files", pluginId],
      });
    },
  });
}

export function usePushPlugin() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ pluginId, message }: { pluginId: string; message: string }) =>
      api.pushPlugin(pluginId, message),
    onSuccess: (_data, { pluginId }) => {
      queryClient.invalidateQueries({
        queryKey: ["user-plugins", pluginId, "git-status"],
      });
    },
  });
}
