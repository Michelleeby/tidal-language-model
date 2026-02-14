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
} from "@tidal/shared";
import { getAuthToken, requestAuth } from "../hooks/useAuth.js";

const BASE = "/api";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const token = getAuthToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const res = await fetch(url, { ...init, headers });

  if (res.status === 401) {
    requestAuth();
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
};
