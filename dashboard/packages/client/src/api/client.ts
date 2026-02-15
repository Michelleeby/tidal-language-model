import type {
  ExperimentsResponse,
  MetricsResponse,
  RLMetricsResponse,
  StatusResponse,
  CheckpointsResponse,
  EvaluationResponse,
  AblationResponse,
  GenerateRequest,
  GenerateResponse,
  CreateJobRequest,
  CreateJobResponse,
  JobsListResponse,
  JobResponse,
  JobSignalRequest,
  JobSignalResponse,
  JobLogsResponse,
  PluginsListResponse,
  PluginResponse,
  ConfigFileResponse,
  ConfigListResponse,
  ReportsListResponse,
  ReportResponse,
  CreateReportRequest,
  UpdateReportRequest,
  DeleteReportResponse,
  UserPluginsListResponse,
  UserPluginResponse,
  CreateUserPluginRequest,
  UpdateUserPluginRequest,
  DeleteUserPluginResponse,
  PluginFileTreeResponse,
  PluginFileReadResponse,
  PluginFileWriteRequest,
} from "@tidal/shared";

const BASE = "/api";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    credentials: "include",
  });

  if (res.status === 401) {
    // Redirect to login page
    window.location.href = "/";
    throw new Error("Authentication required");
  }

  if (res.status === 429) {
    const retryAfter = res.headers.get("Retry-After") ?? "?";
    throw new Error(`Rate limited. Try again in ${retryAfter} seconds.`);
  }

  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  getExperiments: () =>
    fetchJson<ExperimentsResponse>(`${BASE}/experiments`),

  getMetrics: (expId: string, mode = "recent", window = 5000, maxPoints = 2000) =>
    fetchJson<MetricsResponse>(
      `${BASE}/experiments/${expId}/metrics?mode=${mode}&window=${window}&maxPoints=${maxPoints}`,
    ),

  getRLMetrics: (expId: string) =>
    fetchJson<RLMetricsResponse>(`${BASE}/experiments/${expId}/rl-metrics`),

  getStatus: (expId: string) =>
    fetchJson<StatusResponse>(`${BASE}/experiments/${expId}/status`),

  getCheckpoints: (expId: string) =>
    fetchJson<CheckpointsResponse>(`${BASE}/experiments/${expId}/checkpoints`),

  getEvaluation: (expId: string) =>
    fetchJson<EvaluationResponse>(`${BASE}/experiments/${expId}/evaluation`),

  getAblation: (expId: string) =>
    fetchJson<AblationResponse>(`${BASE}/experiments/${expId}/ablation`),

  generate: (body: GenerateRequest) =>
    fetchJson<GenerateResponse>(`${BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  getJobs: () => fetchJson<JobsListResponse>(`${BASE}/jobs`),

  getJob: (jobId: string) =>
    fetchJson<JobResponse>(`${BASE}/jobs/${jobId}`),

  getActiveJob: () => fetchJson<JobResponse>(`${BASE}/jobs/active`),

  createJob: (body: CreateJobRequest) =>
    fetchJson<CreateJobResponse>(`${BASE}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  signalJob: (jobId: string, signal: JobSignalRequest["signal"]) =>
    fetchJson<JobSignalResponse>(`${BASE}/jobs/${jobId}/signal`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ signal }),
    }),

  cancelJob: (jobId: string) =>
    fetchJson<JobSignalResponse>(`${BASE}/jobs/${jobId}/cancel`, {
      method: "POST",
    }),

  getJobLogs: (jobId: string, offset = 0, limit = 5000) =>
    fetchJson<JobLogsResponse>(
      `${BASE}/jobs/${jobId}/logs?offset=${offset}&limit=${limit}`,
    ),

  getPlugins: () => fetchJson<PluginsListResponse>(`${BASE}/plugins`),

  getPlugin: (name: string) =>
    fetchJson<PluginResponse>(`${BASE}/plugins/${name}`),

  getConfigFiles: (pluginName: string) =>
    fetchJson<ConfigListResponse>(`${BASE}/plugins/${pluginName}/configs`),

  getConfigFile: (pluginName: string, filename: string) =>
    fetchJson<ConfigFileResponse>(
      `${BASE}/plugins/${pluginName}/configs/${encodeURIComponent(filename)}`,
    ),

  // Reports
  getReports: () =>
    fetchJson<ReportsListResponse>(`${BASE}/reports`),

  getReport: (id: string) =>
    fetchJson<ReportResponse>(`${BASE}/reports/${id}`),

  createReport: (body?: CreateReportRequest) =>
    fetchJson<ReportResponse>(`${BASE}/reports`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    }),

  updateReport: (id: string, body: UpdateReportRequest) =>
    fetchJson<ReportResponse>(`${BASE}/reports/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  deleteReport: (id: string) =>
    fetchJson<DeleteReportResponse>(`${BASE}/reports/${id}`, {
      method: "DELETE",
    }),

  // User Plugins
  getUserPlugins: () =>
    fetchJson<UserPluginsListResponse>(`${BASE}/user-plugins`),

  getUserPlugin: (id: string) =>
    fetchJson<UserPluginResponse>(`${BASE}/user-plugins/${id}`),

  createUserPlugin: (body: CreateUserPluginRequest) =>
    fetchJson<UserPluginResponse>(`${BASE}/user-plugins`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  updateUserPlugin: (id: string, body: UpdateUserPluginRequest) =>
    fetchJson<UserPluginResponse>(`${BASE}/user-plugins/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  deleteUserPlugin: (id: string) =>
    fetchJson<DeleteUserPluginResponse>(`${BASE}/user-plugins/${id}`, {
      method: "DELETE",
    }),

  getPluginFileTree: (pluginId: string) =>
    fetchJson<PluginFileTreeResponse>(
      `${BASE}/user-plugins/${pluginId}/files`,
    ),

  getPluginFile: (pluginId: string, filePath: string) =>
    fetchJson<PluginFileReadResponse>(
      `${BASE}/user-plugins/${pluginId}/files/${filePath}`,
    ),

  savePluginFile: (pluginId: string, filePath: string, content: string) =>
    fetchJson<{ path: string; saved: boolean }>(
      `${BASE}/user-plugins/${pluginId}/files/${filePath}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content } satisfies PluginFileWriteRequest),
      },
    ),

  createPluginFile: (pluginId: string, filePath: string, content: string) =>
    fetchJson<{ path: string; created: boolean }>(
      `${BASE}/user-plugins/${pluginId}/files/${filePath}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content } satisfies PluginFileWriteRequest),
      },
    ),

  deletePluginFile: (pluginId: string, filePath: string) =>
    fetchJson<{ path: string; deleted: boolean }>(
      `${BASE}/user-plugins/${pluginId}/files/${filePath}`,
      { method: "DELETE" },
    ),
};
