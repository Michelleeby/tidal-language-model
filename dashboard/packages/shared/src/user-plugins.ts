// ---------------------------------------------------------------------------
// User Plugin types — shared between server and client
// ---------------------------------------------------------------------------

/** Full user plugin record as stored in SQLite. */
export interface UserPlugin {
  id: string;
  userId: string;
  name: string;
  displayName: string;
  createdAt: number; // epoch ms
  updatedAt: number; // epoch ms
}

/** Lightweight summary for list views. */
export interface UserPluginSummary {
  id: string;
  name: string;
  displayName: string;
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
