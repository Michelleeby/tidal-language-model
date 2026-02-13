import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client.js";
import type { PluginManifest } from "@tidal/shared";

/**
 * Fetches the full manifest for the default plugin.
 * Returns the manifest or undefined while loading.
 */
export function usePlugin(): {
  manifest: PluginManifest | undefined;
  isLoading: boolean;
} {
  const { data: listData } = useQuery({
    queryKey: ["plugins"],
    queryFn: () => api.getPlugins(),
    staleTime: 5 * 60_000, // plugins rarely change
  });

  const defaultName = listData?.plugins?.[0]?.name;

  const { data: pluginData, isLoading } = useQuery({
    queryKey: ["plugins", defaultName],
    queryFn: () => api.getPlugin(defaultName!),
    enabled: !!defaultName,
    staleTime: 5 * 60_000,
  });

  return {
    manifest: pluginData?.plugin ?? undefined,
    isLoading,
  };
}
