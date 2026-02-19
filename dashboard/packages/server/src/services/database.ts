import BetterSqlite3 from "better-sqlite3";
import { nanoid } from "nanoid";
import type {
  Report,
  ReportSummary,
  BlockContent,
  User,
  AllowedUser,
  AnalysisResult,
  AnalysisResultSummary,
  AnalysisType,
} from "@tidal/shared";

// ---------------------------------------------------------------------------
// Row types (SQLite stores integers, JSON as TEXT)
// ---------------------------------------------------------------------------

interface UserRow {
  id: string;
  github_id: number;
  github_login: string;
  github_avatar_url: string | null;
  github_access_token: string | null;
  created_at: number;
  last_login_at: number;
}

interface AllowedUserRow {
  id: string;
  github_login: string;
  added_by: string | null;
  created_at: number;
}

interface ReportRow {
  id: string;
  user_id: string | null;
  title: string;
  blocks: string; // JSON text
  created_at: number;
  updated_at: number;
}

interface AnalysisResultRow {
  id: string;
  experiment_id: string;
  analysis_type: string;
  label: string;
  request: string; // JSON text
  data: string; // JSON text
  size_bytes: number;
  created_at: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function userRowToUser(row: UserRow): User {
  return {
    id: row.id,
    githubId: row.github_id,
    githubLogin: row.github_login,
    githubAvatarUrl: row.github_avatar_url,
    githubAccessToken: row.github_access_token,
    createdAt: row.created_at,
    lastLoginAt: row.last_login_at,
  };
}

function allowedUserRowToAllowedUser(row: AllowedUserRow): AllowedUser {
  return {
    id: row.id,
    githubLogin: row.github_login,
    addedBy: row.added_by,
    createdAt: row.created_at,
  };
}

function analysisRowToResult(row: AnalysisResultRow): AnalysisResult {
  return {
    id: row.id,
    experimentId: row.experiment_id,
    analysisType: row.analysis_type as AnalysisType,
    label: row.label,
    request: JSON.parse(row.request) as Record<string, unknown>,
    data: JSON.parse(row.data) as Record<string, unknown>,
    sizeBytes: row.size_bytes,
    createdAt: row.created_at,
  };
}

function analysisRowToSummary(row: AnalysisResultRow): AnalysisResultSummary {
  return {
    id: row.id,
    experimentId: row.experiment_id,
    analysisType: row.analysis_type as AnalysisType,
    label: row.label,
    sizeBytes: row.size_bytes,
    createdAt: row.created_at,
  };
}

function reportRowToReport(row: ReportRow): Report {
  return {
    id: row.id,
    userId: row.user_id,
    title: row.title,
    blocks: JSON.parse(row.blocks) as BlockContent[],
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

function reportRowToSummary(row: ReportRow): ReportSummary {
  return {
    id: row.id,
    userId: row.user_id,
    title: row.title,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

// ---------------------------------------------------------------------------
// Database service
// ---------------------------------------------------------------------------

export class Database {
  private db: BetterSqlite3.Database;

  // Prepared statements (lazily created after schema init)
  private stmts!: ReturnType<Database["prepareStatements"]>;

  constructor(dbPath: string) {
    this.db = new BetterSqlite3(dbPath);
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("foreign_keys = ON");
    this.initSchema();
    this.stmts = this.prepareStatements();
  }

  private initSchema(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        github_id INTEGER UNIQUE NOT NULL,
        github_login TEXT NOT NULL,
        github_avatar_url TEXT,
        github_access_token TEXT,
        created_at INTEGER NOT NULL,
        last_login_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS reports (
        id TEXT PRIMARY KEY,
        user_id TEXT REFERENCES users(id),
        title TEXT NOT NULL DEFAULT 'Untitled Report',
        blocks TEXT NOT NULL DEFAULT '[]',
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS allowed_users (
        id TEXT PRIMARY KEY,
        github_login TEXT UNIQUE NOT NULL COLLATE NOCASE,
        added_by TEXT,
        created_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS analysis_results (
        id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        analysis_type TEXT NOT NULL,
        label TEXT NOT NULL,
        request TEXT NOT NULL DEFAULT '{}',
        data TEXT NOT NULL DEFAULT '{}',
        size_bytes INTEGER NOT NULL DEFAULT 0,
        created_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_analysis_results_experiment
        ON analysis_results(experiment_id);
    `);

    // Migrations for existing databases
    this.migrateAddColumn("users", "github_access_token", "TEXT");
  }

  private migrateAddColumn(table: string, column: string, type: string): void {
    const info = this.db.pragma(`table_info(${table})`) as Array<{ name: string }>;
    if (!info.some((col) => col.name === column)) {
      this.db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${type}`);
    }
  }

  private prepareStatements() {
    return {
      upsertUser: this.db.prepare(`
        INSERT INTO users (id, github_id, github_login, github_avatar_url, github_access_token, created_at, last_login_at)
        VALUES (@id, @githubId, @githubLogin, @githubAvatarUrl, @githubAccessToken, @now, @now)
        ON CONFLICT(github_id) DO UPDATE SET
          github_login = @githubLogin,
          github_avatar_url = @githubAvatarUrl,
          github_access_token = @githubAccessToken,
          last_login_at = @now
      `),
      getUserById: this.db.prepare("SELECT * FROM users WHERE id = ?"),
      getUserByGithubId: this.db.prepare("SELECT * FROM users WHERE github_id = ?"),
      getUserByGithubLogin: this.db.prepare("SELECT * FROM users WHERE github_login = ?"),
      listUsers: this.db.prepare("SELECT * FROM users ORDER BY last_login_at DESC"),

      createReport: this.db.prepare(`
        INSERT INTO reports (id, user_id, title, blocks, created_at, updated_at)
        VALUES (@id, @userId, @title, @blocks, @now, @now)
      `),
      getReport: this.db.prepare("SELECT * FROM reports WHERE id = ?"),
      listReports: this.db.prepare("SELECT * FROM reports ORDER BY updated_at DESC"),
      listReportsByUser: this.db.prepare("SELECT * FROM reports WHERE user_id = ? ORDER BY updated_at DESC"),
      updateReportTitle: this.db.prepare("UPDATE reports SET title = @title, updated_at = @now WHERE id = @id"),
      updateReportBlocks: this.db.prepare("UPDATE reports SET blocks = @blocks, updated_at = @now WHERE id = @id"),
      updateReportBoth: this.db.prepare("UPDATE reports SET title = @title, blocks = @blocks, updated_at = @now WHERE id = @id"),
      deleteReport: this.db.prepare("DELETE FROM reports WHERE id = ?"),

      importLegacyReport: this.db.prepare(`
        INSERT OR IGNORE INTO reports (id, user_id, title, blocks, created_at, updated_at)
        VALUES (@id, NULL, @title, @blocks, @createdAt, @updatedAt)
      `),

      addAllowedUser: this.db.prepare(`
        INSERT OR IGNORE INTO allowed_users (id, github_login, added_by, created_at)
        VALUES (@id, @githubLogin, @addedBy, @now)
      `),
      removeAllowedUser: this.db.prepare(
        "DELETE FROM allowed_users WHERE github_login = ? COLLATE NOCASE",
      ),
      getAllowedUser: this.db.prepare(
        "SELECT * FROM allowed_users WHERE github_login = ? COLLATE NOCASE",
      ),
      listAllowedUsers: this.db.prepare(
        "SELECT * FROM allowed_users ORDER BY created_at ASC",
      ),
      countAllowedUsers: this.db.prepare(
        "SELECT COUNT(*) AS cnt FROM allowed_users",
      ),

      // Analysis results
      createAnalysis: this.db.prepare(`
        INSERT INTO analysis_results (id, experiment_id, analysis_type, label, request, data, size_bytes, created_at)
        VALUES (@id, @experimentId, @analysisType, @label, @request, @data, @sizeBytes, @now)
      `),
      getAnalysis: this.db.prepare("SELECT * FROM analysis_results WHERE id = ?"),
      listAnalysesByExperiment: this.db.prepare(
        "SELECT * FROM analysis_results WHERE experiment_id = ? ORDER BY created_at DESC",
      ),
      deleteAnalysis: this.db.prepare("DELETE FROM analysis_results WHERE id = ?"),
    };
  }

  // -------------------------------------------------------------------------
  // User operations
  // -------------------------------------------------------------------------

  upsertUser(params: {
    githubId: number;
    githubLogin: string;
    githubAvatarUrl: string | null;
    githubAccessToken?: string | null;
  }): User {
    const now = Date.now();
    // Check if user already exists to preserve their internal id
    const existing = this.stmts.getUserByGithubId.get(params.githubId) as UserRow | undefined;
    const id = existing?.id ?? nanoid();

    this.stmts.upsertUser.run({
      id,
      githubId: params.githubId,
      githubLogin: params.githubLogin,
      githubAvatarUrl: params.githubAvatarUrl,
      githubAccessToken: params.githubAccessToken ?? null,
      now,
    });

    return userRowToUser(this.stmts.getUserByGithubId.get(params.githubId) as UserRow);
  }

  getUserById(id: string): User | null {
    const row = this.stmts.getUserById.get(id) as UserRow | undefined;
    return row ? userRowToUser(row) : null;
  }

  getUserByGithubId(githubId: number): User | null {
    const row = this.stmts.getUserByGithubId.get(githubId) as UserRow | undefined;
    return row ? userRowToUser(row) : null;
  }

  getUserByGithubLogin(login: string): User | null {
    const row = this.stmts.getUserByGithubLogin.get(login) as UserRow | undefined;
    return row ? userRowToUser(row) : null;
  }

  listUsers(): User[] {
    const rows = this.stmts.listUsers.all() as UserRow[];
    return rows.map(userRowToUser);
  }

  // -------------------------------------------------------------------------
  // Report operations
  // -------------------------------------------------------------------------

  createReport(title?: string, userId?: string): Report {
    const now = Date.now();
    const id = nanoid();

    this.stmts.createReport.run({
      id,
      userId: userId ?? null,
      title: title ?? "Untitled Report",
      blocks: "[]",
      now,
    });

    return reportRowToReport(this.stmts.getReport.get(id) as ReportRow);
  }

  getReport(id: string): Report | null {
    const row = this.stmts.getReport.get(id) as ReportRow | undefined;
    return row ? reportRowToReport(row) : null;
  }

  listReports(): ReportSummary[] {
    const rows = this.stmts.listReports.all() as ReportRow[];
    return rows.map(reportRowToSummary);
  }

  listReportsByUser(userId: string): ReportSummary[] {
    const rows = this.stmts.listReportsByUser.all(userId) as ReportRow[];
    return rows.map(reportRowToSummary);
  }

  updateReport(
    id: string,
    patch: { title?: string; blocks?: BlockContent[] },
  ): Report | null {
    const existing = this.stmts.getReport.get(id) as ReportRow | undefined;
    if (!existing) return null;

    const now = Date.now();

    if (patch.title !== undefined && patch.blocks !== undefined) {
      this.stmts.updateReportBoth.run({
        id,
        title: patch.title,
        blocks: JSON.stringify(patch.blocks),
        now,
      });
    } else if (patch.title !== undefined) {
      this.stmts.updateReportTitle.run({ id, title: patch.title, now });
    } else if (patch.blocks !== undefined) {
      this.stmts.updateReportBlocks.run({
        id,
        blocks: JSON.stringify(patch.blocks),
        now,
      });
    }

    return reportRowToReport(this.stmts.getReport.get(id) as ReportRow);
  }

  deleteReport(id: string): boolean {
    const result = this.stmts.deleteReport.run(id);
    return result.changes > 0;
  }

  // -------------------------------------------------------------------------
  // Allowed users (whitelist)
  // -------------------------------------------------------------------------

  addAllowedUser(githubLogin: string, addedBy: string | null): AllowedUser | null {
    const now = Date.now();
    const id = nanoid();

    const result = this.stmts.addAllowedUser.run({
      id,
      githubLogin,
      addedBy,
      now,
    });

    if (result.changes === 0) return null; // duplicate

    return allowedUserRowToAllowedUser(
      this.stmts.getAllowedUser.get(githubLogin) as AllowedUserRow,
    );
  }

  removeAllowedUser(githubLogin: string): boolean {
    const result = this.stmts.removeAllowedUser.run(githubLogin);
    return result.changes > 0;
  }

  isUserAllowed(githubLogin: string): boolean {
    const row = this.stmts.getAllowedUser.get(githubLogin) as AllowedUserRow | undefined;
    return !!row;
  }

  listAllowedUsers(): AllowedUser[] {
    const rows = this.stmts.listAllowedUsers.all() as AllowedUserRow[];
    return rows.map(allowedUserRowToAllowedUser);
  }

  countAllowedUsers(): number {
    const row = this.stmts.countAllowedUsers.get() as { cnt: number };
    return row.cnt;
  }

  // -------------------------------------------------------------------------
  // Analysis results
  // -------------------------------------------------------------------------

  createAnalysis(params: {
    experimentId: string;
    analysisType: AnalysisType;
    label: string;
    request: Record<string, unknown>;
    data: Record<string, unknown>;
  }): AnalysisResult {
    const now = Date.now();
    const id = nanoid();
    const requestJson = JSON.stringify(params.request);
    const dataJson = JSON.stringify(params.data);
    const sizeBytes = Buffer.byteLength(dataJson, "utf-8");

    this.stmts.createAnalysis.run({
      id,
      experimentId: params.experimentId,
      analysisType: params.analysisType,
      label: params.label,
      request: requestJson,
      data: dataJson,
      sizeBytes,
      now,
    });

    return analysisRowToResult(
      this.stmts.getAnalysis.get(id) as AnalysisResultRow,
    );
  }

  getAnalysis(id: string): AnalysisResult | null {
    const row = this.stmts.getAnalysis.get(id) as AnalysisResultRow | undefined;
    return row ? analysisRowToResult(row) : null;
  }

  listAnalyses(experimentId: string): AnalysisResultSummary[] {
    const rows = this.stmts.listAnalysesByExperiment.all(
      experimentId,
    ) as AnalysisResultRow[];
    return rows.map(analysisRowToSummary);
  }

  deleteAnalysis(id: string): boolean {
    const result = this.stmts.deleteAnalysis.run(id);
    return result.changes > 0;
  }

  // -------------------------------------------------------------------------
  // Legacy migration
  // -------------------------------------------------------------------------

  importLegacyReport(report: {
    id: string;
    title: string;
    blocks: BlockContent[];
    createdAt: number;
    updatedAt: number;
  }): boolean {
    const result = this.stmts.importLegacyReport.run({
      id: report.id,
      title: report.title,
      blocks: JSON.stringify(report.blocks),
      createdAt: report.createdAt,
      updatedAt: report.updatedAt,
    });
    return result.changes > 0;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  close(): void {
    this.db.close();
  }
}
