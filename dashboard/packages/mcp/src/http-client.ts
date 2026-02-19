// ---------------------------------------------------------------------------
// HTTP client abstraction for the Tidal Dashboard API
// ---------------------------------------------------------------------------

/** Successful API result. */
export interface ApiOk<T> {
  ok: true;
  data: T;
  status: number;
}

/** Failed API result. */
export interface ApiErr {
  ok: false;
  error: string;
  status: number;
}

/** Discriminated union returned by all client methods. */
export type ApiResult<T> = ApiOk<T> | ApiErr;

/** Injectable interface so tests can mock the HTTP layer. */
export interface TidalApiClient {
  get<T>(
    path: string,
    query?: Record<string, string | number | undefined>,
  ): Promise<ApiResult<T>>;

  post<T>(path: string, body: unknown): Promise<ApiResult<T>>;
}

// ---------------------------------------------------------------------------
// Caching decorator — caches GET responses for completed experiments to disk
// ---------------------------------------------------------------------------

import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";

/** Regex matching per-experiment API paths: /api/experiments/:expId/... */
const EXP_PATH_RE = /^\/api\/experiments\/([^/]+)\/(.+)$/;

/** Cached response stored on disk. */
interface CacheEntry {
  cacheKey: string;
  status: number;
  data: unknown;
  cachedAt: string;
}

export class CachingTidalApiClient implements TidalApiClient {
  private completedExperiments = new Set<string>();

  constructor(
    private inner: TidalApiClient,
    private cacheDir: string,
  ) {}

  async get<T>(
    path: string,
    query?: Record<string, string | number | undefined>,
  ): Promise<ApiResult<T>> {
    const expId = this.parseExpId(path);
    const cacheKey = this.buildCacheKey(path, query);

    // Try reading from disk cache if experiment is known-completed
    if (expId && this.completedExperiments.has(expId)) {
      const cached = await this.readCache(expId, cacheKey);
      if (cached) {
        return { ok: true, data: cached.data as T, status: cached.status };
      }
    }

    // Delegate to inner client
    const result = await this.inner.get<T>(path, query);

    // Inspect responses to learn completion status
    this.inspectForCompletion(path, result);

    // Write to cache if successful and experiment is completed
    if (result.ok && expId && this.completedExperiments.has(expId)) {
      await this.writeCache(expId, cacheKey, result.status, result.data);
    }

    return result;
  }

  async post<T>(path: string, body: unknown): Promise<ApiResult<T>> {
    return this.inner.post<T>(path, body);
  }

  // ── Cache key ─────────────────────────────────────────────────────

  private buildCacheKey(
    path: string,
    query?: Record<string, string | number | undefined>,
  ): string {
    let key = path;
    if (query) {
      const params = new URLSearchParams();
      for (const [k, v] of Object.entries(query)) {
        if (v !== undefined) params.set(k, String(v));
      }
      params.sort();
      const qs = params.toString();
      if (qs) key += `?${qs}`;
    }
    return key;
  }

  private hashKey(cacheKey: string): string {
    return createHash("sha256").update(cacheKey).digest("hex").slice(0, 16);
  }

  // ── Path parsing ──────────────────────────────────────────────────

  private parseExpId(path: string): string | null {
    const m = EXP_PATH_RE.exec(path);
    return m ? m[1] : null;
  }

  // ── Completion detection ──────────────────────────────────────────

  private inspectForCompletion<T>(path: string, result: ApiResult<T>): void {
    if (!result.ok) return;

    // list_experiments → scan for completed experiments
    if (path === "/api/experiments") {
      const data = result.data as {
        experiments?: Array<{
          expId: string;
          status?: { status?: string };
        }>;
      };
      if (data.experiments) {
        for (const exp of data.experiments) {
          if (exp.status?.status === "completed") {
            this.completedExperiments.add(exp.expId);
          }
        }
      }
      return;
    }

    // get_status → check single experiment
    const m = EXP_PATH_RE.exec(path);
    if (m && m[2] === "status") {
      const data = result.data as { status?: string };
      if (data.status === "completed") {
        this.completedExperiments.add(m[1]);
      }
    }
  }

  // ── Disk I/O ──────────────────────────────────────────────────────

  private cacheFilePath(expId: string, cacheKey: string): string {
    return join(this.cacheDir, expId, `${this.hashKey(cacheKey)}.json`);
  }

  private async readCache(
    expId: string,
    cacheKey: string,
  ): Promise<CacheEntry | null> {
    try {
      const raw = await readFile(this.cacheFilePath(expId, cacheKey), "utf-8");
      const entry = JSON.parse(raw) as CacheEntry;
      // Verify cacheKey matches to guard against hash collisions
      if (entry.cacheKey !== cacheKey) return null;
      return entry;
    } catch {
      return null;
    }
  }

  private async writeCache(
    expId: string,
    cacheKey: string,
    status: number,
    data: unknown,
  ): Promise<void> {
    try {
      const filePath = this.cacheFilePath(expId, cacheKey);
      await mkdir(join(this.cacheDir, expId), { recursive: true });
      const entry: CacheEntry = {
        cacheKey,
        status,
        data,
        cachedAt: new Date().toISOString(),
      };
      await writeFile(filePath, JSON.stringify(entry));
    } catch {
      // Silently degrade — disk full, read-only, etc.
    }
  }
}

// ---------------------------------------------------------------------------
// Production implementation using Node 20's built-in fetch
// ---------------------------------------------------------------------------

export class FetchTidalApiClient implements TidalApiClient {
  constructor(
    private baseUrl: string,
    private token?: string,
  ) {}

  async get<T>(
    path: string,
    query?: Record<string, string | number | undefined>,
  ): Promise<ApiResult<T>> {
    let url = `${this.baseUrl}${path}`;

    if (query) {
      const params = new URLSearchParams();
      for (const [key, value] of Object.entries(query)) {
        if (value !== undefined) {
          params.set(key, String(value));
        }
      }
      const qs = params.toString();
      if (qs) url += `?${qs}`;
    }

    return this.execute<T>(url, { method: "GET", headers: this.headers() });
  }

  async post<T>(path: string, body: unknown): Promise<ApiResult<T>> {
    const url = `${this.baseUrl}${path}`;
    return this.execute<T>(url, {
      method: "POST",
      headers: { ...this.headers(), "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = {};
    if (this.token) {
      h["Authorization"] = `Bearer ${this.token}`;
    }
    return h;
  }

  private async execute<T>(
    url: string,
    init: RequestInit,
  ): Promise<ApiResult<T>> {
    try {
      const res = await fetch(url, init);
      if (res.ok) {
        const data = (await res.json()) as T;
        return { ok: true, data, status: res.status };
      }
      const text = await res.text();
      let error: string;
      try {
        const parsed = JSON.parse(text);
        error = parsed.error ?? parsed.message ?? text;
      } catch {
        error = text;
      }
      return { ok: false, error, status: res.status };
    } catch (err) {
      return {
        ok: false,
        error: `Network error: ${err instanceof Error ? err.message : String(err)}`,
        status: 0,
      };
    }
  }
}
