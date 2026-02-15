import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client.js";
import type { CreateUserPluginRequest } from "@tidal/shared";

export function useUserPlugins() {
  return useQuery({
    queryKey: ["user-plugins"],
    queryFn: () => api.getUserPlugins(),
  });
}

export function useUserPlugin(id: string | null) {
  return useQuery({
    queryKey: ["user-plugins", id],
    queryFn: () => api.getUserPlugin(id!),
    enabled: !!id,
  });
}

export function useCreateUserPlugin() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateUserPluginRequest) => api.createUserPlugin(body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["user-plugins"] });
    },
  });
}

export function useUpdateUserPlugin() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, displayName }: { id: string; displayName: string }) =>
      api.updateUserPlugin(id, { displayName }),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["user-plugins", variables.id],
      });
      queryClient.invalidateQueries({ queryKey: ["user-plugins"] });
    },
  });
}

export function useDeleteUserPlugin() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteUserPlugin(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["user-plugins"] });
    },
  });
}
