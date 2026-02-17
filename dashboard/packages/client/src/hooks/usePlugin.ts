import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client.js";
import type { PluginManifest } from "@tidal/shared";

/**
 * Fetches the full manifest for the tidal plugin.
 */
export function usePlugin(): {
  manifest: PluginManifest | undefined;
  isLoading: boolean;
} {
  const { data, isLoading } = useQuery({
    queryKey: ["plugins", "tidal"],
    queryFn: () => api.getPlugin("tidal"),
    staleTime: 5 * 60_000,
  });

  return {
    manifest: data?.plugin ?? undefined,
    isLoading,
  };
}
