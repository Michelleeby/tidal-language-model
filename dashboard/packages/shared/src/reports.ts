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
  /** True when the report has been explicitly saved to Spaces (Phase 4). */
  savedToSpaces?: boolean;
}

/** Lightweight summary for list views (no blocks payload). */
export interface ReportSummary {
  id: string;
  userId: string | null;
  title: string;
  createdAt: number;
  updatedAt: number;
  /** True when the report has local changes not yet saved to Spaces (Phase 4). */
  isDraft?: boolean;
}

// ---------------------------------------------------------------------------
// API request / response types
// ---------------------------------------------------------------------------

export interface ReportsListResponse {
  reports: ReportSummary[];
  spacesAvailable?: boolean;
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

/** POST /api/reports/:id/save — explicit Spaces save (Phase 4). */
export interface SaveReportRequest {
  title?: string;
  blocks?: BlockContent[];
}

export interface SaveReportResponse {
  report: Report;
}

/** GET /api/reports/:id/versions (Phase 4). */
export interface ReportVersion {
  timestamp: number;
  spacesKey: string;
}

export interface ReportVersionsResponse {
  versions: ReportVersion[];
}

/** POST /api/reports/:id/restore (Phase 4). */
export interface RestoreVersionRequest {
  timestamp: number;
}

// ---------------------------------------------------------------------------
// Generate report (block-pattern based)
// ---------------------------------------------------------------------------

export interface GenerateReportRequest {
  pattern: string;
  experimentId: string;
  title?: string;
  githubLogin?: string;
}

export interface GenerateReportResponse {
  report: Report;
}
