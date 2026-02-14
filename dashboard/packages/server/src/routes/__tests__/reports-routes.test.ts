import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import Fastify from "fastify";
import type { FastifyInstance } from "fastify";
import type { ServerConfig } from "../../config.js";
import authPlugin from "../../plugins/auth.js";
import reportsRoutes from "../reports.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TEST_TOKEN = "test-secret-token";
const AUTH_HEADER = `Bearer ${TEST_TOKEN}`;

const cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "reports-route-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

async function buildApp(projectRoot: string): Promise<FastifyInstance> {
  const app = Fastify({ logger: false });
  app.decorate("serverConfig", {
    projectRoot,
    authToken: TEST_TOKEN,
  } as unknown as ServerConfig);
  await app.register(authPlugin);
  await app.register(reportsRoutes);
  return app;
}

/** Helper: create a report with auth, return the report object. */
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

// ---------------------------------------------------------------------------
// Auth enforcement
// ---------------------------------------------------------------------------

describe("Reports auth enforcement", () => {
  it("returns 401 for POST without token", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

    const resp = await app.inject({
      method: "POST",
      url: "/api/reports",
      payload: {},
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });

  it("returns 401 for PUT without token", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

    const resp = await app.inject({
      method: "PUT",
      url: "/api/reports/some-id",
      payload: { title: "X" },
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });

  it("returns 401 for DELETE without token", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

    const resp = await app.inject({
      method: "DELETE",
      url: "/api/reports/some-id",
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });

  it("allows GET /api/reports without token", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

    const resp = await app.inject({
      method: "GET",
      url: "/api/reports",
    });

    assert.equal(resp.statusCode, 200);
    await app.close();
  });

  it("allows GET /api/reports/:id without token", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

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
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

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
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

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
});

// ---------------------------------------------------------------------------
// GET /api/reports
// ---------------------------------------------------------------------------

describe("GET /api/reports", () => {
  it("returns empty list when no reports exist", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

    const resp = await app.inject({
      method: "GET",
      url: "/api/reports",
    });

    assert.equal(resp.statusCode, 200);
    assert.deepEqual(resp.json().reports, []);

    await app.close();
  });

  it("returns created reports", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

    await createReport(app, "A");
    await createReport(app, "B");

    const resp = await app.inject({ method: "GET", url: "/api/reports" });

    assert.equal(resp.statusCode, 200);
    const { reports } = resp.json();
    assert.equal(reports.length, 2);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// GET /api/reports/:id
// ---------------------------------------------------------------------------

describe("GET /api/reports/:id", () => {
  it("returns a report by id", async () => {
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

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
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

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
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

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
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

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
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

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
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

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
    const tmpDir = await freshTmpDir();
    const app = await buildApp(tmpDir);

    const resp = await app.inject({
      method: "DELETE",
      url: "/api/reports/nonexistent",
      headers: { authorization: AUTH_HEADER },
    });

    assert.equal(resp.statusCode, 404);

    await app.close();
  });
});
