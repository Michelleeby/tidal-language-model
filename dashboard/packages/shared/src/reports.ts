// ---------------------------------------------------------------------------
// Report types — shared between server and client
// ---------------------------------------------------------------------------

/** A single block in the BlockNote editor (opaque JSON). */
export type BlockContent = Record<string, unknown>;

/** Full report as stored on disk / returned by GET /api/reports/:id. */
export interface Report {
  id: string;
  userId: string | null;
  title: string;
  blocks: BlockContent[];
  createdAt: number; // epoch ms
  updatedAt: number; // epoch ms
}

/** Lightweight summary for list views (no blocks payload). */
export interface ReportSummary {
  id: string;
  userId: string | null;
  title: string;
  createdAt: number;
  updatedAt: number;
}

// ---------------------------------------------------------------------------
// API request / response types
// ---------------------------------------------------------------------------

export interface ReportsListResponse {
  reports: ReportSummary[];
}

export interface ReportResponse {
  report: Report;
}

export interface CreateReportRequest {
  title?: string;
}

export interface UpdateReportRequest {
  title?: string;
  blocks?: BlockContent[];
}

export interface DeleteReportResponse {
  deleted: boolean;
}
