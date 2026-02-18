import { useState, useCallback, useEffect } from "react";
import type { AllowedUser } from "@tidal/shared";

export function useAllowedUsers() {
  const [allowedUsers, setAllowedUsers] = useState<AllowedUser[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAllowedUsers = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/admin/allowed-users", {
        credentials: "include",
      });
      if (!res.ok) throw new Error("Failed to fetch allowed users");
      const data = await res.json();
      setAllowedUsers(data.allowedUsers);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  const addUser = useCallback(async (githubLogin: string) => {
    setError(null);
    try {
      const res = await fetch("/api/admin/allowed-users", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ githubLogin }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error ?? "Failed to add user");
      }
      await fetchAllowedUsers();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  }, [fetchAllowedUsers]);

  const removeUser = useCallback(async (githubLogin: string) => {
    setError(null);
    try {
      const res = await fetch(`/api/admin/allowed-users/${encodeURIComponent(githubLogin)}`, {
        method: "DELETE",
        credentials: "include",
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error ?? "Failed to remove user");
      }
      await fetchAllowedUsers();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  }, [fetchAllowedUsers]);

  useEffect(() => {
    fetchAllowedUsers();
  }, [fetchAllowedUsers]);

  return { allowedUsers, loading, error, addUser, removeUser, refresh: fetchAllowedUsers };
}
