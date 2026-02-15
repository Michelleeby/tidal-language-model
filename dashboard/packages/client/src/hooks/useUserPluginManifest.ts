import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client.js";

export function useUserPluginManifest(pluginId: string | null) {
  return useQuery({
    queryKey: ["user-plugins", pluginId, "manifest"],
    queryFn: () => api.getUserPluginManifest(pluginId!),
    enabled: !!pluginId,
    staleTime: 60_000,
  });
}
