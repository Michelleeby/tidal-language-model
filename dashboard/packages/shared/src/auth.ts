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

// ---------------------------------------------------------------------------
// Allowed users (whitelist)
// ---------------------------------------------------------------------------

/** A GitHub user on the access whitelist. */
export interface AllowedUser {
  id: string;
  githubLogin: string;
  addedBy: string | null;
  createdAt: number; // epoch ms
}

/** GET /api/admin/allowed-users response. */
export interface AllowedUsersResponse {
  allowedUsers: AllowedUser[];
}

/** POST /api/admin/allowed-users request body. */
export interface AddAllowedUserRequest {
  githubLogin: string;
}

/** POST /api/admin/allowed-users response. */
export interface AddAllowedUserResponse {
  allowedUser: AllowedUser;
  created: boolean;
}

/** DELETE /api/admin/allowed-users/:githubLogin response. */
export interface RemoveAllowedUserResponse {
  removed: boolean;
}
