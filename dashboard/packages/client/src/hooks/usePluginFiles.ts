import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client.js";

export function usePluginFileTree(pluginId: string | null) {
  return useQuery({
    queryKey: ["plugin-files", pluginId, "tree"],
    queryFn: () => api.getPluginFileTree(pluginId!),
    enabled: !!pluginId,
  });
}

export function usePluginFile(pluginId: string | null, filePath: string | null) {
  return useQuery({
    queryKey: ["plugin-files", pluginId, filePath],
    queryFn: () => api.getPluginFile(pluginId!, filePath!),
    enabled: !!pluginId && !!filePath,
  });
}

export function useSavePluginFile() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      pluginId,
      filePath,
      content,
    }: {
      pluginId: string;
      filePath: string;
      content: string;
    }) => api.savePluginFile(pluginId, filePath, content),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({
        queryKey: [
          "plugin-files",
          variables.pluginId,
          variables.filePath,
        ],
      });
    },
  });
}
