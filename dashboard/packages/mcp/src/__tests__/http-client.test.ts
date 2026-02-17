import { describe, it, beforeEach, afterEach, mock } from "node:test";
import assert from "node:assert/strict";
import { FetchTidalApiClient } from "../http-client.js";

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
