import type { MetricPoint, TrainingStatus, RLTrainingMetrics } from "./metrics.js";
import type { TrainingJob, LogLine } from "./jobs.js";

/** SSE event types sent by GET /api/experiments/:expId/stream */
export type SSEEvent =
  | { type: "metrics"; data: MetricPoint }
  | { type: "status"; data: TrainingStatus }
  | { type: "rl-metrics"; data: RLTrainingMetrics }
  | { type: "heartbeat"; data: { timestamp: number } }
  | { type: "job-update"; data: TrainingJob }
  | { type: "log-lines"; data: { jobId: string; lines: LogLine[] } };

export type SSEEventType = SSEEvent["type"];
