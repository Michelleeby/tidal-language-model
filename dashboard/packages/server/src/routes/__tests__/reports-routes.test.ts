import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import Fastify from "fastify";
import cookie from "@fastify/cookie";
import { SignJWT } from "jose";
import type { FastifyInstance } from "fastify";
import type { ServerConfig } from "../../config.js";
import { Database } from "../../services/database.js";
import { ObjectStore } from "../../services/object-store.js";
import authPlugin from "../../plugins/auth.js";
import reportsRoutes from "../reports.js";
import type { GenerateReportRequest } from "@tidal/shared";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TEST_TOKEN = "test-secret-token";
const JWT_SECRET = "test-jwt-secret-at-least-32-chars-long!";
const AUTH_HEADER = `Bearer ${TEST_TOKEN}`;

const cleanups: Array<() => Promise<void>> = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "reports-route-test-"));
  cleanups.push(async () => {
    await fsp.rm(dir, { recursive: true, force: true });
  });
  return dir;
}

after(async () => {
  for (const fn of cleanups) await fn();
});

async function buildApp(
  objectStore?: ObjectStore,
): Promise<{ app: FastifyInstance; db: Database }> {
  const dir = await freshTmpDir();
  const db = new Database(path.join(dir, "test.db"));

  const app = Fastify({ logger: false });
  app.decorate("serverConfig", {
    projectRoot: dir,
    authToken: TEST_TOKEN,
    jwtSecret: JWT_SECRET,
  } as unknown as ServerConfig);
  app.decorate("db", db);
  app.decorate("objectStore", objectStore ?? new ObjectStore(null));

  await app.register(cookie);
  await app.register(authPlugin);
  await app.register(reportsRoutes);

  cleanups.push(async () => {
    db.close();
  });

  return { app, db };
}

/** Helper: create a report with Bearer auth, return the report object. */
async function createReport(app: FastifyInstance, title?: string) {
  const resp = await app.inject({
    method: "POST",
    url: "/api/reports",
    headers: { authorization: AUTH_HEADER },
    payload: title ? { title } : {},
  });
  assert.equal(resp.statusCode, 201);
  return resp.json().report;
}

async function createJwt(payload: Record<string, unknown>): Promise<string> {
  const key = new TextEncoder().encode(JWT_SECRET);
  return new SignJWT(payload)
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime("1h")
    .sign(key);
}

// ---------------------------------------------------------------------------
// Auth enforcement
// ---------------------------------------------------------------------------

describe("Reports auth enforcement", () => {
  it("returns 401 for POST without token", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports",
      payload: {},
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });

  it("returns 401 for PUT without token", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "PUT",
      url: "/api/reports/some-id",
      payload: { title: "X" },
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });

  it("returns 401 for DELETE without token", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "DELETE",
      url: "/api/reports/some-id",
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });

  it("allows GET /api/reports without token", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "GET",
      url: "/api/reports",
    });

    assert.equal(resp.statusCode, 200);
    await app.close();
  });

  it("allows GET /api/reports/:id without token", async () => {
    const { app } = await buildApp();

    const report = await createReport(app);

    const resp = await app.inject({
      method: "GET",
      url: `/api/reports/${report.id}`,
    });

    assert.equal(resp.statusCode, 200);
    await app.close();
  });
});

// ---------------------------------------------------------------------------
// POST /api/reports
// ---------------------------------------------------------------------------

describe("POST /api/reports", () => {
  it("creates a report with default title", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports",
      headers: { authorization: AUTH_HEADER },
      payload: {},
    });

    assert.equal(resp.statusCode, 201);
    const body = resp.json();
    assert.ok(body.report.id);
    assert.equal(body.report.title, "Untitled Report");
    assert.deepEqual(body.report.blocks, []);

    await app.close();
  });

  it("creates a report with a custom title", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports",
      headers: { authorization: AUTH_HEADER },
      payload: { title: "My Report" },
    });

    assert.equal(resp.statusCode, 201);
    assert.equal(resp.json().report.title, "My Report");

    await app.close();
  });

  it("sets userId from JWT cookie", async () => {
    const { app, db } = await buildApp();

    const user = db.upsertUser({
      githubId: 42,
      githubLogin: "jwtuser",
      githubAvatarUrl: null,
    });

    const token = await createJwt({ sub: user.id, githubLogin: "jwtuser" });

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports",
      cookies: { tidal_session: token },
      payload: { title: "JWT Report" },
    });

    assert.equal(resp.statusCode, 201);
    const body = resp.json();
    assert.equal(body.report.userId, user.id);

    await app.close();
  });

  it("sets userId to null for Bearer auth", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports",
      headers: { authorization: AUTH_HEADER },
      payload: { title: "Bearer Report" },
    });

    assert.equal(resp.statusCode, 201);
    assert.equal(resp.json().report.userId, null);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// GET /api/reports
// ---------------------------------------------------------------------------

describe("GET /api/reports", () => {
  it("returns empty list when no reports exist", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "GET",
      url: "/api/reports",
    });

    assert.equal(resp.statusCode, 200);
    assert.deepEqual(resp.json().reports, []);

    await app.close();
  });

  it("returns created reports", async () => {
    const { app } = await buildApp();

    await createReport(app, "A");
    await createReport(app, "B");

    const resp = await app.inject({ method: "GET", url: "/api/reports" });

    assert.equal(resp.statusCode, 200);
    const { reports } = resp.json();
    assert.equal(reports.length, 2);

    await app.close();
  });

  it("includes userId in summaries", async () => {
    const { app } = await buildApp();

    await createReport(app, "Test");

    const resp = await app.inject({ method: "GET", url: "/api/reports" });
    const { reports } = resp.json();

    assert.equal(reports[0].userId, null);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// GET /api/reports/:id
// ---------------------------------------------------------------------------

describe("GET /api/reports/:id", () => {
  it("returns a report by id", async () => {
    const { app } = await buildApp();

    const report = await createReport(app, "Test");

    const resp = await app.inject({
      method: "GET",
      url: `/api/reports/${report.id}`,
    });

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.json().report.id, report.id);
    assert.equal(resp.json().report.title, "Test");

    await app.close();
  });

  it("returns 404 for non-existent id", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "GET",
      url: "/api/reports/nonexistent",
    });

    assert.equal(resp.statusCode, 404);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// PUT /api/reports/:id
// ---------------------------------------------------------------------------

describe("PUT /api/reports/:id", () => {
  it("updates report title", async () => {
    const { app } = await buildApp();

    const report = await createReport(app);

    const resp = await app.inject({
      method: "PUT",
      url: `/api/reports/${report.id}`,
      headers: { authorization: AUTH_HEADER },
      payload: { title: "Updated" },
    });

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.json().report.title, "Updated");

    await app.close();
  });

  it("updates report blocks", async () => {
    const { app } = await buildApp();

    const report = await createReport(app);

    const blocks = [{ type: "paragraph", content: [{ type: "text", text: "Hello" }] }];
    const resp = await app.inject({
      method: "PUT",
      url: `/api/reports/${report.id}`,
      headers: { authorization: AUTH_HEADER },
      payload: { blocks },
    });

    assert.equal(resp.statusCode, 200);
    assert.deepEqual(resp.json().report.blocks, blocks);

    await app.close();
  });

  it("returns 404 for non-existent id", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "PUT",
      url: "/api/reports/nonexistent",
      headers: { authorization: AUTH_HEADER },
      payload: { title: "X" },
    });

    assert.equal(resp.statusCode, 404);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// DELETE /api/reports/:id
// ---------------------------------------------------------------------------

describe("DELETE /api/reports/:id", () => {
  it("deletes an existing report", async () => {
    const { app } = await buildApp();

    const report = await createReport(app);

    const delResp = await app.inject({
      method: "DELETE",
      url: `/api/reports/${report.id}`,
      headers: { authorization: AUTH_HEADER },
    });

    assert.equal(delResp.statusCode, 200);
    assert.equal(delResp.json().deleted, true);

    // Confirm it's gone
    const getResp = await app.inject({
      method: "GET",
      url: `/api/reports/${report.id}`,
    });
    assert.equal(getResp.statusCode, 404);

    await app.close();
  });

  it("returns 404 for non-existent id", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "DELETE",
      url: "/api/reports/nonexistent",
      headers: { authorization: AUTH_HEADER },
    });

    assert.equal(resp.statusCode, 404);

    await app.close();
  });

  it("cleans up Spaces objects when objectStore is configured", async () => {
    const deletedPrefixes: string[] = [];
    const mockClient = { send: async () => ({}) };
    const mockCommands = {
      PutObject: class { constructor(public input: Record<string, unknown>) {} },
      DeleteObject: class { constructor(public input: Record<string, unknown>) {} },
      DeleteObjects: class { constructor(public input: Record<string, unknown>) {} },
      ListObjectsV2: class { constructor(public input: Record<string, unknown>) {} },
      HeadObject: class { constructor(public input: Record<string, unknown>) {} },
      GetObject: class { constructor(public input: Record<string, unknown>) {} },
    };

    const store = new ObjectStore(
      { endpoint: "https://test.example.com", region: "us-east-1", accessKeyId: "key", secretAccessKey: "secret", bucket: "test-bucket" },
      mockClient,
      mockCommands as any,
    );

    // Spy on deletePrefix
    const originalDeletePrefix = store.deletePrefix.bind(store);
    store.deletePrefix = async (prefix: string) => {
      deletedPrefixes.push(prefix);
      return originalDeletePrefix(prefix);
    };

    const { app } = await buildApp(store);

    const report = await createReport(app);

    const delResp = await app.inject({
      method: "DELETE",
      url: `/api/reports/${report.id}`,
      headers: { authorization: AUTH_HEADER },
    });

    assert.equal(delResp.statusCode, 200);
    assert.equal(delResp.json().deleted, true);

    // Verify Spaces cleanup was called with the correct prefix
    assert.equal(deletedPrefixes.length, 1);
    assert.equal(deletedPrefixes[0], `reports/${report.id}/`);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// GET /api/reports — spacesAvailable field
// ---------------------------------------------------------------------------

describe("GET /api/reports — spacesAvailable", () => {
  it("returns spacesAvailable: false when objectStore is not configured", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({ method: "GET", url: "/api/reports" });

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.json().spacesAvailable, false);

    await app.close();
  });

  it("returns spacesAvailable: true when objectStore is configured", async () => {
    const mockClient = { send: async () => ({}) };
    const mockCommands = {
      PutObject: class { constructor(public input: Record<string, unknown>) {} },
      DeleteObject: class { constructor(public input: Record<string, unknown>) {} },
      DeleteObjects: class { constructor(public input: Record<string, unknown>) {} },
      ListObjectsV2: class { constructor(public input: Record<string, unknown>) {} },
      HeadObject: class { constructor(public input: Record<string, unknown>) {} },
      GetObject: class { constructor(public input: Record<string, unknown>) {} },
    };
    const store = new ObjectStore(
      { endpoint: "https://test.example.com", region: "us-east-1", accessKeyId: "key", secretAccessKey: "secret", bucket: "test-bucket" },
      mockClient,
      mockCommands as any,
    );

    const { app } = await buildApp(store);

    const resp = await app.inject({ method: "GET", url: "/api/reports" });

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.json().spacesAvailable, true);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// POST /api/reports/:id/save
// ---------------------------------------------------------------------------

describe("POST /api/reports/:id/save", () => {
  it("returns 401 without auth", async () => {
    const { app } = await buildApp();
    const report = await createReport(app);

    const resp = await app.inject({
      method: "POST",
      url: `/api/reports/${report.id}/save`,
      payload: { title: "Saved" },
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });

  it("saves and returns the report", async () => {
    const { app } = await buildApp();
    const report = await createReport(app, "Original");

    const resp = await app.inject({
      method: "POST",
      url: `/api/reports/${report.id}/save`,
      headers: { authorization: AUTH_HEADER },
      payload: { title: "Saved Title" },
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.ok(body.report);
    assert.equal(body.report.title, "Saved Title");
    assert.equal(body.spacesWritten, false, "Spaces not configured in test env");

    await app.close();
  });

  it("returns 404 for non-existent report", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports/nonexistent/save",
      headers: { authorization: AUTH_HEADER },
      payload: {},
    });

    assert.equal(resp.statusCode, 404);
    await app.close();
  });
});

// ---------------------------------------------------------------------------
// GET /api/reports/:id/versions
// ---------------------------------------------------------------------------

describe("GET /api/reports/:id/versions", () => {
  it("returns empty versions array when Spaces not configured", async () => {
    const { app } = await buildApp();
    const report = await createReport(app);

    const resp = await app.inject({
      method: "GET",
      url: `/api/reports/${report.id}/versions`,
    });

    assert.equal(resp.statusCode, 200);
    assert.deepEqual(resp.json().versions, []);

    await app.close();
  });

  it("does not require auth", async () => {
    const { app } = await buildApp();
    const report = await createReport(app);

    const resp = await app.inject({
      method: "GET",
      url: `/api/reports/${report.id}/versions`,
    });

    assert.equal(resp.statusCode, 200);
    await app.close();
  });
});

// ---------------------------------------------------------------------------
// POST /api/reports/:id/restore
// ---------------------------------------------------------------------------

describe("POST /api/reports/:id/restore", () => {
  it("returns 401 without auth", async () => {
    const { app } = await buildApp();
    const report = await createReport(app);

    const resp = await app.inject({
      method: "POST",
      url: `/api/reports/${report.id}/restore`,
      payload: { timestamp: 12345 },
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });

  it("returns 404 when Spaces not configured (nothing to restore)", async () => {
    const { app } = await buildApp();
    const report = await createReport(app);

    const resp = await app.inject({
      method: "POST",
      url: `/api/reports/${report.id}/restore`,
      headers: { authorization: AUTH_HEADER },
      payload: { timestamp: 12345 },
    });

    assert.equal(resp.statusCode, 404);
    await app.close();
  });
});

// ---------------------------------------------------------------------------
// POST /api/reports/generate
// ---------------------------------------------------------------------------

describe("POST /api/reports/generate", () => {
  it("returns 401 without auth", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports/generate",
      payload: { pattern: "experiment-overview", experimentId: "exp-1" },
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });

  it("returns 400 for unknown pattern", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports/generate",
      headers: { authorization: AUTH_HEADER },
      payload: { pattern: "nonexistent", experimentId: "exp-1" },
    });

    assert.equal(resp.statusCode, 400);
    assert.match(resp.json().error, /Unknown pattern/);
    await app.close();
  });

  it("returns 400 for missing experimentId", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports/generate",
      headers: { authorization: AUTH_HEADER },
      payload: { pattern: "experiment-overview" },
    });

    assert.equal(resp.statusCode, 400);
    assert.match(resp.json().error, /experimentId/);
    await app.close();
  });

  it("creates report with experiment-overview pattern", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports/generate",
      headers: { authorization: AUTH_HEADER },
      payload: { pattern: "experiment-overview", experimentId: "exp-123" } satisfies GenerateReportRequest,
    });

    assert.equal(resp.statusCode, 201);
    const { report } = resp.json();
    assert.ok(report.id);
    assert.ok(report.blocks.length > 0);
    // Verify experimentId appears in block props
    const chartBlock = report.blocks.find((b: Record<string, unknown>) => b.type === "experimentChart");
    assert.ok(chartBlock);
    assert.equal((chartBlock as { props: { experimentId: string } }).props.experimentId, "exp-123");
    await app.close();
  });

  it("creates report with rl-analysis pattern", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports/generate",
      headers: { authorization: AUTH_HEADER },
      payload: { pattern: "rl-analysis", experimentId: "exp-1" },
    });

    assert.equal(resp.statusCode, 201);
    assert.ok(resp.json().report.blocks.length > 0);
    await app.close();
  });

  it("creates report with trajectory-report pattern", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports/generate",
      headers: { authorization: AUTH_HEADER },
      payload: { pattern: "trajectory-report", experimentId: "exp-1" },
    });

    assert.equal(resp.statusCode, 201);
    assert.ok(resp.json().report.blocks.length > 0);
    await app.close();
  });

  it("creates report with full-report pattern", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports/generate",
      headers: { authorization: AUTH_HEADER },
      payload: { pattern: "full-report", experimentId: "exp-1" },
    });

    assert.equal(resp.statusCode, 201);
    // full-report is all three patterns combined + a top heading
    const { report } = resp.json();
    assert.ok(report.blocks.length > 10);
    await app.close();
  });

  it("uses custom title when provided", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports/generate",
      headers: { authorization: AUTH_HEADER },
      payload: { pattern: "experiment-overview", experimentId: "exp-1", title: "My Custom Title" },
    });

    assert.equal(resp.statusCode, 201);
    assert.equal(resp.json().report.title, "My Custom Title");
    await app.close();
  });

  it("associates user via githubLogin", async () => {
    const { app, db } = await buildApp();

    const user = db.upsertUser({
      githubId: 42,
      githubLogin: "octocat",
      githubAvatarUrl: null,
    });

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports/generate",
      headers: { authorization: AUTH_HEADER },
      payload: { pattern: "experiment-overview", experimentId: "exp-1", githubLogin: "octocat" },
    });

    assert.equal(resp.statusCode, 201);
    assert.equal(resp.json().report.userId, user.id);
    await app.close();
  });

  it("gracefully sets null userId for unknown githubLogin", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports/generate",
      headers: { authorization: AUTH_HEADER },
      payload: { pattern: "experiment-overview", experimentId: "exp-1", githubLogin: "nobody" },
    });

    assert.equal(resp.statusCode, 201);
    assert.equal(resp.json().report.userId, null);
    await app.close();
  });
});
