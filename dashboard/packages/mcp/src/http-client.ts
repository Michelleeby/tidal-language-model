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
