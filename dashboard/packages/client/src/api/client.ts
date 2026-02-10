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
} from "@tidal/shared";

const BASE = "/api";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
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
};
