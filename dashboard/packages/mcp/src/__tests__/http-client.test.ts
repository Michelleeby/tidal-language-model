import { describe, it, beforeEach, afterEach, mock } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { writeFile, mkdir } from "node:fs/promises";
import {
  FetchTidalApiClient,
  CachingTidalApiClient,
  type TidalApiClient,
  type ApiResult,
} from "../http-client.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function jsonResponse(data: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => data,
    text: async () => JSON.stringify(data),
  } as Response;
}

function errorResponse(status: number, message: string): Response {
  return {
    ok: false,
    status,
    json: async () => ({ error: message }),
    text: async () => JSON.stringify({ error: message }),
  } as Response;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("FetchTidalApiClient", () => {
  let originalFetch: typeof globalThis.fetch;
  let mockFetch: ReturnType<typeof mock.fn<typeof globalThis.fetch>>;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    mockFetch = mock.fn<typeof globalThis.fetch>();
    globalThis.fetch = mockFetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  // ── GET ────────────────────────────────────────────────────────────

  describe("get()", () => {
    it("constructs URL with base and path", async () => {
      mockFetch.mock.mockImplementation(async () =>
        jsonResponse({ experiments: [] }),
      );
      const client = new FetchTidalApiClient("http://localhost:4400");

      await client.get("/api/experiments");

      assert.equal(mockFetch.mock.callCount(), 1);
      const [url] = mockFetch.mock.calls[0].arguments;
      assert.equal(url, "http://localhost:4400/api/experiments");
    });

    it("appends query parameters to URL", async () => {
      mockFetch.mock.mockImplementation(async () =>
        jsonResponse({ expId: "abc", points: [] }),
      );
      const client = new FetchTidalApiClient("http://localhost:4400");

      await client.get("/api/experiments/abc/metrics", {
        mode: "recent",
        window: 100,
      });

      const [url] = mockFetch.mock.calls[0].arguments;
      assert.equal(
        url,
        "http://localhost:4400/api/experiments/abc/metrics?mode=recent&window=100",
      );
    });

    it("skips undefined query values", async () => {
      mockFetch.mock.mockImplementation(async () => jsonResponse({}));
      const client = new FetchTidalApiClient("http://localhost:4400");

      await client.get("/api/test", { a: "1", b: undefined, c: "3" });

      const [url] = mockFetch.mock.calls[0].arguments;
      assert.equal(url, "http://localhost:4400/api/test?a=1&c=3");
    });

    it("returns ok result with data on 2xx", async () => {
      const data = { experiments: [{ id: "exp1" }] };
      mockFetch.mock.mockImplementation(async () => jsonResponse(data));
      const client = new FetchTidalApiClient("http://localhost:4400");

      const result = await client.get("/api/experiments");

      assert.equal(result.ok, true);
      if (result.ok) {
        assert.deepEqual(result.data, data);
        assert.equal(result.status, 200);
      }
    });

    it("returns error result on 4xx", async () => {
      mockFetch.mock.mockImplementation(async () =>
        errorResponse(404, "Not found"),
      );
      const client = new FetchTidalApiClient("http://localhost:4400");

      const result = await client.get("/api/experiments/missing/metrics");

      assert.equal(result.ok, false);
      if (!result.ok) {
        assert.equal(result.status, 404);
        assert.match(result.error, /Not found/);
      }
    });

    it("returns error result on 5xx", async () => {
      mockFetch.mock.mockImplementation(async () =>
        errorResponse(500, "Internal server error"),
      );
      const client = new FetchTidalApiClient("http://localhost:4400");

      const result = await client.get("/api/experiments");

      assert.equal(result.ok, false);
      if (!result.ok) {
        assert.equal(result.status, 500);
      }
    });

    it("returns network error with status 0 on fetch failure", async () => {
      mockFetch.mock.mockImplementation(async () => {
        throw new Error("ECONNREFUSED");
      });
      const client = new FetchTidalApiClient("http://localhost:4400");

      const result = await client.get("/api/experiments");

      assert.equal(result.ok, false);
      if (!result.ok) {
        assert.equal(result.status, 0);
        assert.match(result.error, /Network error/);
        assert.match(result.error, /ECONNREFUSED/);
      }
    });
  });

  // ── POST ───────────────────────────────────────────────────────────

  describe("post()", () => {
    it("sends JSON body with correct headers", async () => {
      const responseData = { text: "Once upon a time..." };
      mockFetch.mock.mockImplementation(async () =>
        jsonResponse(responseData),
      );
      const client = new FetchTidalApiClient("http://localhost:4400");

      const body = { checkpoint: "model.pth", prompt: "Hello" };
      await client.post("/api/generate", body);

      assert.equal(mockFetch.mock.callCount(), 1);
      const [url, init] = mockFetch.mock.calls[0].arguments;
      assert.equal(url, "http://localhost:4400/api/generate");
      assert.equal(init?.method, "POST");
      assert.equal(
        (init?.headers as Record<string, string>)["Content-Type"],
        "application/json",
      );
      assert.equal(init?.body, JSON.stringify(body));
    });

    it("returns ok result with data on 2xx", async () => {
      const data = { text: "generated text", tokensGenerated: 10 };
      mockFetch.mock.mockImplementation(async () => jsonResponse(data));
      const client = new FetchTidalApiClient("http://localhost:4400");

      const result = await client.post("/api/generate", {
        checkpoint: "x",
        prompt: "y",
      });

      assert.equal(result.ok, true);
      if (result.ok) {
        assert.deepEqual(result.data, data);
      }
    });

    it("returns error result on 4xx", async () => {
      mockFetch.mock.mockImplementation(async () =>
        errorResponse(400, "Missing checkpoint"),
      );
      const client = new FetchTidalApiClient("http://localhost:4400");

      const result = await client.post("/api/generate", { prompt: "y" });

      assert.equal(result.ok, false);
      if (!result.ok) {
        assert.equal(result.status, 400);
      }
    });

    it("returns network error on fetch failure", async () => {
      mockFetch.mock.mockImplementation(async () => {
        throw new TypeError("fetch failed");
      });
      const client = new FetchTidalApiClient("http://localhost:4400");

      const result = await client.post("/api/generate", {});

      assert.equal(result.ok, false);
      if (!result.ok) {
        assert.equal(result.status, 0);
        assert.match(result.error, /Network error/);
      }
    });
  });

  // ── PUT ────────────────────────────────────────────────────────────

  describe("put()", () => {
    it("sends JSON body with PUT method", async () => {
      const responseData = { report: { id: "rpt-1" } };
      mockFetch.mock.mockImplementation(async () =>
        jsonResponse(responseData),
      );
      const client = new FetchTidalApiClient("http://localhost:4400");

      const body = { title: "Updated", blocks: [{ type: "heading" }] };
      await client.put("/api/reports/rpt-1", body);

      assert.equal(mockFetch.mock.callCount(), 1);
      const [url, init] = mockFetch.mock.calls[0].arguments;
      assert.equal(url, "http://localhost:4400/api/reports/rpt-1");
      assert.equal(init?.method, "PUT");
      assert.equal(
        (init?.headers as Record<string, string>)["Content-Type"],
        "application/json",
      );
      assert.equal(init?.body, JSON.stringify(body));
    });

    it("returns ok result with data on 2xx", async () => {
      const data = { report: { id: "rpt-1", title: "Updated" } };
      mockFetch.mock.mockImplementation(async () => jsonResponse(data));
      const client = new FetchTidalApiClient("http://localhost:4400");

      const result = await client.put("/api/reports/rpt-1", {
        title: "Updated",
      });

      assert.equal(result.ok, true);
      if (result.ok) {
        assert.deepEqual(result.data, data);
      }
    });

    it("returns error result on 4xx", async () => {
      mockFetch.mock.mockImplementation(async () =>
        errorResponse(404, "Report not found"),
      );
      const client = new FetchTidalApiClient("http://localhost:4400");

      const result = await client.put("/api/reports/missing", {
        title: "x",
      });

      assert.equal(result.ok, false);
      if (!result.ok) {
        assert.equal(result.status, 404);
        assert.match(result.error, /Report not found/);
      }
    });

    it("returns network error on fetch failure", async () => {
      mockFetch.mock.mockImplementation(async () => {
        throw new TypeError("fetch failed");
      });
      const client = new FetchTidalApiClient("http://localhost:4400");

      const result = await client.put("/api/reports/rpt-1", {});

      assert.equal(result.ok, false);
      if (!result.ok) {
        assert.equal(result.status, 0);
        assert.match(result.error, /Network error/);
      }
    });
  });

  // ── Auth ───────────────────────────────────────────────────────────

  describe("auth token", () => {
    it("sends Bearer token when configured", async () => {
      mockFetch.mock.mockImplementation(async () => jsonResponse({}));
      const client = new FetchTidalApiClient(
        "http://localhost:4400",
        "my-secret-token",
      );

      await client.get("/api/experiments");

      const [, init] = mockFetch.mock.calls[0].arguments;
      assert.equal(
        (init?.headers as Record<string, string>)["Authorization"],
        "Bearer my-secret-token",
      );
    });

    it("does not send Authorization header when no token", async () => {
      mockFetch.mock.mockImplementation(async () => jsonResponse({}));
      const client = new FetchTidalApiClient("http://localhost:4400");

      await client.get("/api/experiments");

      const [, init] = mockFetch.mock.calls[0].arguments;
      assert.equal(
        (init?.headers as Record<string, string>)["Authorization"],
        undefined,
      );
    });
  });
});

// ---------------------------------------------------------------------------
// CachingTidalApiClient
// ---------------------------------------------------------------------------

/** Creates a spy client that records calls and returns path-specific responses. */
function spyClient(
  responses: Map<string, ApiResult<unknown>>,
): TidalApiClient & { getCalls: string[]; postCalls: string[]; putCalls: string[] } {
  const getCalls: string[] = [];
  const postCalls: string[] = [];
  const putCalls: string[] = [];

  return {
    getCalls,
    postCalls,
    putCalls,
    async get<T>(
      path: string,
      query?: Record<string, string | number | undefined>,
    ): Promise<ApiResult<T>> {
      // Build a canonical key that includes sorted query params
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
      getCalls.push(key);
      const response = responses.get(key) ?? responses.get(path);
      if (!response) {
        return { ok: false, error: "No mock for " + key, status: 404 } as ApiResult<T>;
      }
      return response as ApiResult<T>;
    },
    async post<T>(path: string, _body: unknown): Promise<ApiResult<T>> {
      postCalls.push(path);
      const response = responses.get(path);
      if (!response) {
        return { ok: false, error: "No mock for " + path, status: 404 } as ApiResult<T>;
      }
      return response as ApiResult<T>;
    },
    async put<T>(path: string, _body: unknown): Promise<ApiResult<T>> {
      putCalls.push(path);
      const response = responses.get(path);
      if (!response) {
        return { ok: false, error: "No mock for " + path, status: 404 } as ApiResult<T>;
      }
      return response as ApiResult<T>;
    },
  };
}

describe("CachingTidalApiClient", () => {
  let cacheDir: string;

  beforeEach(async () => {
    cacheDir = await mkdtemp(join(tmpdir(), "tidal-cache-test-"));
  });

  afterEach(async () => {
    await rm(cacheDir, { recursive: true, force: true });
  });

  // ── 1. post() always delegates, never caches ────────────────────────

  it("post() always delegates and never caches", async () => {
    const responses = new Map<string, ApiResult<unknown>>([
      ["/api/generate", { ok: true, data: { text: "hello" }, status: 200 }],
    ]);
    const inner = spyClient(responses);
    const client = new CachingTidalApiClient(inner, cacheDir);

    await client.post("/api/generate", { prompt: "a" });
    await client.post("/api/generate", { prompt: "a" });

    assert.equal(inner.postCalls.length, 2);
    // No cache files written
    const files = await readdir(cacheDir, { recursive: true });
    const jsonFiles = files.filter((f: string) => f.endsWith(".json"));
    assert.equal(jsonFiles.length, 0);
  });

  // ── 1b. put() always delegates, never caches ───────────────────────

  it("put() always delegates and never caches", async () => {
    const responses = new Map<string, ApiResult<unknown>>([
      ["/api/reports/rpt-1", { ok: true, data: { report: { id: "rpt-1" } }, status: 200 }],
    ]);
    const inner = spyClient(responses);
    const client = new CachingTidalApiClient(inner, cacheDir);

    await client.put("/api/reports/rpt-1", { title: "a" });
    await client.put("/api/reports/rpt-1", { title: "b" });

    assert.equal(inner.putCalls.length, 2);
    // No cache files written
    const files = await readdir(cacheDir, { recursive: true });
    const jsonFiles = files.filter((f: string) => f.endsWith(".json"));
    assert.equal(jsonFiles.length, 0);
  });

  // ── 2. get() for non-experiment paths always delegates ──────────────

  it("get() for non-experiment paths always delegates", async () => {
    const responses = new Map<string, ApiResult<unknown>>([
      ["/api/plugins", { ok: true, data: { plugins: [] }, status: 200 }],
    ]);
    const inner = spyClient(responses);
    const client = new CachingTidalApiClient(inner, cacheDir);

    await client.get("/api/plugins");
    await client.get("/api/plugins");

    assert.equal(inner.getCalls.length, 2);
  });

  // ── 3. get() for experiment list always delegates ───────────────────

  it("get() for experiment list always delegates", async () => {
    const responses = new Map<string, ApiResult<unknown>>([
      [
        "/api/experiments",
        {
          ok: true,
          data: {
            experiments: [
              { expId: "exp1", status: { status: "completed" } },
            ],
          },
          status: 200,
        },
      ],
    ]);
    const inner = spyClient(responses);
    const client = new CachingTidalApiClient(inner, cacheDir);

    await client.get("/api/experiments");
    await client.get("/api/experiments");

    // Both calls hit the inner client (not cached)
    assert.equal(inner.getCalls.length, 2);
  });

  // ── 4. get() for unknown-status experiment always delegates ─────────

  it("get() for unknown-status experiment always delegates", async () => {
    const responses = new Map<string, ApiResult<unknown>>([
      [
        "/api/experiments/exp1/metrics",
        { ok: true, data: { points: [1, 2, 3] }, status: 200 },
      ],
    ]);
    const inner = spyClient(responses);
    const client = new CachingTidalApiClient(inner, cacheDir);

    // No list_experiments or get_status call, so exp1 status is unknown
    await client.get("/api/experiments/exp1/metrics");
    await client.get("/api/experiments/exp1/metrics");

    assert.equal(inner.getCalls.length, 2);
  });

  // ── 5. list_experiments marks completed experiments ──────────────────

  it("list_experiments marks completed experiments and enables caching", async () => {
    const metricsData = { points: [1, 2, 3] };
    const responses = new Map<string, ApiResult<unknown>>([
      [
        "/api/experiments",
        {
          ok: true,
          data: {
            experiments: [
              { expId: "exp1", status: { status: "completed" } },
              { expId: "exp2", status: { status: "training" } },
            ],
          },
          status: 200,
        },
      ],
      [
        "/api/experiments/exp1/metrics",
        { ok: true, data: metricsData, status: 200 },
      ],
    ]);
    const inner = spyClient(responses);
    const client = new CachingTidalApiClient(inner, cacheDir);

    // First: learn completion status
    await client.get("/api/experiments");

    // Fetch metrics for completed exp1 — first call goes to network
    const r1 = await client.get("/api/experiments/exp1/metrics");
    assert.equal(r1.ok, true);
    if (r1.ok) assert.deepEqual(r1.data, metricsData);

    // Second call should come from cache (no additional inner call)
    const r2 = await client.get("/api/experiments/exp1/metrics");
    assert.equal(r2.ok, true);
    if (r2.ok) assert.deepEqual(r2.data, metricsData);

    // inner client: 1 list + 1 metrics (second metrics was cached)
    assert.equal(inner.getCalls.length, 2);
  });

  // ── 6. get_status marks completed experiment ────────────────────────

  it("get_status marks completed experiment and enables caching", async () => {
    const statusData = { status: "completed", currentEpoch: 3, totalEpochs: 3 };
    const metricsData = { points: [10, 20] };
    const responses = new Map<string, ApiResult<unknown>>([
      [
        "/api/experiments/exp1/status",
        { ok: true, data: statusData, status: 200 },
      ],
      [
        "/api/experiments/exp1/metrics",
        { ok: true, data: metricsData, status: 200 },
      ],
    ]);
    const inner = spyClient(responses);
    const client = new CachingTidalApiClient(inner, cacheDir);

    // Learn completion via get_status
    await client.get("/api/experiments/exp1/status");

    // Fetch metrics twice — second should be cached
    await client.get("/api/experiments/exp1/metrics");
    await client.get("/api/experiments/exp1/metrics");

    // inner: 1 status + 1 metrics (second metrics cached)
    assert.equal(inner.getCalls.length, 2);
  });

  // ── 7. in-progress experiments are never cached ─────────────────────

  it("in-progress experiments are never cached", async () => {
    const responses = new Map<string, ApiResult<unknown>>([
      [
        "/api/experiments",
        {
          ok: true,
          data: {
            experiments: [
              { expId: "exp1", status: { status: "training" } },
            ],
          },
          status: 200,
        },
      ],
      [
        "/api/experiments/exp1/metrics",
        { ok: true, data: { points: [] }, status: 200 },
      ],
    ]);
    const inner = spyClient(responses);
    const client = new CachingTidalApiClient(inner, cacheDir);

    await client.get("/api/experiments");
    await client.get("/api/experiments/exp1/metrics");
    await client.get("/api/experiments/exp1/metrics");

    // All 3 calls hit inner (list + 2 metrics)
    assert.equal(inner.getCalls.length, 3);
  });

  // ── 8. cache persists across client instances ───────────────────────

  it("cache persists across client instances", async () => {
    const metricsData = { points: [1, 2, 3] };
    const responses = new Map<string, ApiResult<unknown>>([
      [
        "/api/experiments",
        {
          ok: true,
          data: {
            experiments: [
              { expId: "exp1", status: { status: "completed" } },
            ],
          },
          status: 200,
        },
      ],
      [
        "/api/experiments/exp1/metrics",
        { ok: true, data: metricsData, status: 200 },
      ],
    ]);
    const inner1 = spyClient(responses);
    const client1 = new CachingTidalApiClient(inner1, cacheDir);

    // Client 1 learns completion and fetches metrics (populates cache)
    await client1.get("/api/experiments");
    await client1.get("/api/experiments/exp1/metrics");

    // Client 2 with same cacheDir — learns completion then reads cache
    const inner2 = spyClient(responses);
    const client2 = new CachingTidalApiClient(inner2, cacheDir);

    await client2.get("/api/experiments");
    const r = await client2.get("/api/experiments/exp1/metrics");

    assert.equal(r.ok, true);
    if (r.ok) assert.deepEqual(r.data, metricsData);

    // inner2: 1 list call only — metrics came from disk cache
    assert.equal(inner2.getCalls.length, 1);
  });

  // ── 9. different query params produce different cache entries ────────

  it("different query params produce different cache entries", async () => {
    const recentData = { points: [1] };
    const historicalData = { points: [1, 2, 3, 4, 5] };
    const responses = new Map<string, ApiResult<unknown>>([
      [
        "/api/experiments",
        {
          ok: true,
          data: {
            experiments: [
              { expId: "exp1", status: { status: "completed" } },
            ],
          },
          status: 200,
        },
      ],
      [
        "/api/experiments/exp1/metrics?mode=recent",
        { ok: true, data: recentData, status: 200 },
      ],
      [
        "/api/experiments/exp1/metrics?mode=historical",
        { ok: true, data: historicalData, status: 200 },
      ],
    ]);
    const inner = spyClient(responses);
    const client = new CachingTidalApiClient(inner, cacheDir);

    await client.get("/api/experiments");

    const r1 = await client.get("/api/experiments/exp1/metrics", {
      mode: "recent",
    });
    const r2 = await client.get("/api/experiments/exp1/metrics", {
      mode: "historical",
    });

    assert.equal(r1.ok, true);
    assert.equal(r2.ok, true);
    if (r1.ok) assert.deepEqual(r1.data, recentData);
    if (r2.ok) assert.deepEqual(r2.data, historicalData);

    // Now repeat both — should come from cache
    await client.get("/api/experiments/exp1/metrics", { mode: "recent" });
    await client.get("/api/experiments/exp1/metrics", { mode: "historical" });

    // inner: 1 list + 2 metrics (each param variant fetched once)
    assert.equal(inner.getCalls.length, 3);
  });

  // ── 10. corrupt cache file falls back to network ────────────────────

  it("corrupt cache file falls back to network", async () => {
    const metricsData = { points: [42] };
    const responses = new Map<string, ApiResult<unknown>>([
      [
        "/api/experiments",
        {
          ok: true,
          data: {
            experiments: [
              { expId: "exp1", status: { status: "completed" } },
            ],
          },
          status: 200,
        },
      ],
      [
        "/api/experiments/exp1/metrics",
        { ok: true, data: metricsData, status: 200 },
      ],
    ]);
    const inner = spyClient(responses);
    const client = new CachingTidalApiClient(inner, cacheDir);

    // Learn completion and populate cache
    await client.get("/api/experiments");
    await client.get("/api/experiments/exp1/metrics");

    // Corrupt the cache file
    const expDir = join(cacheDir, "exp1");
    const files = await readdir(expDir);
    const cacheFile = files.find((f: string) => f.endsWith(".json"));
    assert.ok(cacheFile, "Cache file should exist");
    await writeFile(join(expDir, cacheFile!), "NOT VALID JSON{{{{");

    // Should fall back to network
    const r = await client.get("/api/experiments/exp1/metrics");
    assert.equal(r.ok, true);
    if (r.ok) assert.deepEqual(r.data, metricsData);

    // inner: 1 list + 2 metrics (original + re-fetch after corruption)
    assert.equal(inner.getCalls.length, 3);
  });

  // ── 11. missing cache directory is created ──────────────────────────

  it("missing cache directory is created on first write", async () => {
    // Use a nested path that doesn't exist yet
    const nestedDir = join(cacheDir, "deep", "nested", "cache");
    const metricsData = { points: [99] };
    const responses = new Map<string, ApiResult<unknown>>([
      [
        "/api/experiments",
        {
          ok: true,
          data: {
            experiments: [
              { expId: "exp1", status: { status: "completed" } },
            ],
          },
          status: 200,
        },
      ],
      [
        "/api/experiments/exp1/metrics",
        { ok: true, data: metricsData, status: 200 },
      ],
    ]);
    const inner = spyClient(responses);
    const client = new CachingTidalApiClient(inner, nestedDir);

    await client.get("/api/experiments");
    await client.get("/api/experiments/exp1/metrics");

    // Verify cache dir was created and file exists
    const expDir = join(nestedDir, "exp1");
    const files = await readdir(expDir);
    assert.ok(files.some((f: string) => f.endsWith(".json")));
  });

  // ── 12. error responses are never cached ────────────────────────────

  it("error responses are never cached", async () => {
    const responses = new Map<string, ApiResult<unknown>>([
      [
        "/api/experiments",
        {
          ok: true,
          data: {
            experiments: [
              { expId: "exp1", status: { status: "completed" } },
            ],
          },
          status: 200,
        },
      ],
      [
        "/api/experiments/exp1/metrics",
        { ok: false, error: "Internal error", status: 500 },
      ],
    ]);
    const inner = spyClient(responses);
    const client = new CachingTidalApiClient(inner, cacheDir);

    await client.get("/api/experiments");
    await client.get("/api/experiments/exp1/metrics");
    await client.get("/api/experiments/exp1/metrics");

    // Both metrics calls hit inner (error not cached)
    assert.equal(inner.getCalls.length, 3);

    // No cache files written
    const files = await readdir(cacheDir, { recursive: true });
    const jsonFiles = files.filter((f: string) => f.endsWith(".json"));
    assert.equal(jsonFiles.length, 0);
  });
});
