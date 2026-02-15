// ---------------------------------------------------------------------------
// User Plugin types — shared between server and client
// ---------------------------------------------------------------------------

/** Full user plugin record as stored in SQLite. */
export interface UserPlugin {
  id: string;
  userId: string;
  name: string;
  displayName: string;
  githubRepoUrl: string;
  createdAt: number; // epoch ms
  updatedAt: number; // epoch ms
}

/** Lightweight summary for list views. */
export interface UserPluginSummary {
  id: string;
  name: string;
  displayName: string;
  githubRepoUrl: string;
  createdAt: number;
  updatedAt: number;
}

/** A node in the plugin file tree. */
export interface PluginFileNode {
  name: string;
  path: string;
  type: "file" | "directory";
  children?: PluginFileNode[];
}

// ---------------------------------------------------------------------------
// API request / response types
// ---------------------------------------------------------------------------

export interface CreateUserPluginRequest {
  name: string;
  displayName: string;
}

export interface UpdateUserPluginRequest {
  displayName: string;
}

export interface UserPluginsListResponse {
  plugins: UserPluginSummary[];
}

export interface UserPluginResponse {
  plugin: UserPlugin;
}

export interface DeleteUserPluginResponse {
  deleted: boolean;
}

export interface PluginFileTreeResponse {
  files: PluginFileNode[];
}

export interface PluginFileReadResponse {
  path: string;
  content: string;
}

export interface PluginFileWriteRequest {
  content: string;
}

// ---------------------------------------------------------------------------
// Git sync types
// ---------------------------------------------------------------------------

export interface PluginGitStatusResponse {
  dirty: boolean;
  files: string[];
}

export interface PluginGitPullResponse {
  ok: boolean;
}

export interface PluginGitPushRequest {
  message: string;
}

export interface PluginGitPushResponse {
  ok: boolean;
}

// ---------------------------------------------------------------------------
// Manifest types
// ---------------------------------------------------------------------------

export interface PluginManifestResponse {
  manifest: Record<string, unknown>;
}
