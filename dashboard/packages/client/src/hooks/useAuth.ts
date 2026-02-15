import { create } from "zustand";
import type { User } from "@tidal/shared";

interface AuthStore {
  user: User | null;
  loading: boolean;
  setUser: (user: User | null) => void;
  setLoading: (loading: boolean) => void;
}

export const useAuthStore = create<AuthStore>((set) => ({
  user: null,
  loading: true,
  setUser: (user) => set({ user, loading: false }),
  setLoading: (loading) => set({ loading }),
}));

/** Check authentication status by calling /api/auth/me. */
export async function checkAuth(): Promise<User | null> {
  const { setUser } = useAuthStore.getState();
  try {
    const res = await fetch("/api/auth/me", { credentials: "include" });
    if (!res.ok) {
      setUser(null);
      return null;
    }
    const data = await res.json();
    setUser(data.user ?? null);
    return data.user ?? null;
  } catch {
    setUser(null);
    return null;
  }
}

/** Redirect to GitHub OAuth flow. */
export function login(): void {
  window.location.href = "/api/auth/github";
}

/** Log out by calling /api/auth/logout. */
export async function logout(): Promise<void> {
  const { setUser } = useAuthStore.getState();
  try {
    await fetch("/api/auth/logout", {
      method: "POST",
      credentials: "include",
    });
  } finally {
    setUser(null);
  }
}
