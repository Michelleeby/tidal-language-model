// ---------------------------------------------------------------------------
// Auth types — shared between server and client
// ---------------------------------------------------------------------------

/** User profile returned from the server. */
export interface User {
  id: string;
  githubId: number;
  githubLogin: string;
  githubAvatarUrl: string | null;
  githubAccessToken?: string | null;
  createdAt: number; // epoch ms
  lastLoginAt: number; // epoch ms
}

/** GET /api/auth/me response. */
export interface AuthMeResponse {
  user: User | null;
}

/** POST /api/auth/logout response. */
export interface AuthLogoutResponse {
  loggedOut: boolean;
}
