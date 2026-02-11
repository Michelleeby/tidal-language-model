import { create } from "zustand";

const SESSION_KEY = "tidal_auth_token";

interface AuthStore {
  token: string | null;
  showPrompt: boolean;
  setToken: (token: string | null) => void;
  requestAuth: () => void;
  dismissPrompt: () => void;
}

export const useAuthStore = create<AuthStore>((set) => ({
  token: sessionStorage.getItem(SESSION_KEY),
  showPrompt: false,
  setToken: (token) => {
    if (token) {
      sessionStorage.setItem(SESSION_KEY, token);
    } else {
      sessionStorage.removeItem(SESSION_KEY);
    }
    set({ token, showPrompt: false });
  },
  requestAuth: () => set({ showPrompt: true }),
  dismissPrompt: () => set({ showPrompt: false }),
}));

export function getAuthToken(): string | null {
  return useAuthStore.getState().token;
}

export function requestAuth(): void {
  useAuthStore.getState().requestAuth();
}
