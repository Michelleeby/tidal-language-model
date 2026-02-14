import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client.js";
import { usePlugin } from "./usePlugin.js";

export function useConfigFiles() {
  const { manifest } = usePlugin();
  const pluginName = manifest?.name;

  return useQuery({
    queryKey: ["config-files", pluginName],
    queryFn: () => api.getConfigFiles(pluginName!),
    enabled: !!pluginName,
    staleTime: 5 * 60_000,
  });
}

export function useConfigFile(filename: string | null) {
  const { manifest } = usePlugin();
  const pluginName = manifest?.name;

  return useQuery({
    queryKey: ["config-file", pluginName, filename],
    queryFn: () => api.getConfigFile(pluginName!, filename!),
    enabled: !!pluginName && !!filename,
    staleTime: 5 * 60_000,
  });
}
