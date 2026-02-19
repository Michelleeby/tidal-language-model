import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import Fastify from "fastify";
import cookie from "@fastify/cookie";
import type { FastifyInstance } from "fastify";
import type { ServerConfig } from "../../config.js";
import { Database } from "../../services/database.js";
import authPlugin from "../../plugins/auth.js";
import analysesRoutes from "../analyses.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TEST_TOKEN = "test-secret-token";
const JWT_SECRET = "test-jwt-secret-at-least-32-chars-long!";
const AUTH_HEADER = `Bearer ${TEST_TOKEN}`;

const cleanups: Array<() => Promise<void>> = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "analyses-route-test-"));
  cleanups.push(async () => {
    await fsp.rm(dir, { recursive: true, force: true });
  });
  return dir;
}

after(async () => {
  for (const fn of cleanups) await fn();
});

async function buildApp(): Promise<{ app: FastifyInstance; db: Database }> {
  const dir = await freshTmpDir();
  const db = new Database(path.join(dir, "test.db"));

  const app = Fastify({ logger: false });
  app.decorate("serverConfig", {
    projectRoot: dir,
    authToken: TEST_TOKEN,
    jwtSecret: JWT_SECRET,
  } as unknown as ServerConfig);
  app.decorate("db", db);

  await app.register(cookie);
  await app.register(authPlugin);
  await app.register(analysesRoutes);

  cleanups.push(async () => {
    db.close();
  });

  return { app, db };
}

const samplePayload = {
  analysisType: "trajectory" as const,
  label: "Test trajectory",
  request: { checkpoint: "/path/to/ckpt" },
  data: { text: "hello", trajectory: { gateSignals: [[0.5]] } },
};

// ---------------------------------------------------------------------------
// POST /api/experiments/:expId/analyses
// ---------------------------------------------------------------------------

describe("POST /api/experiments/:expId/analyses", () => {
  it("creates an analysis result", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/experiments/exp-123/analyses",
      headers: { authorization: AUTH_HEADER },
      payload: samplePayload,
    });

    assert.equal(resp.statusCode, 201);
    const { analysis } = resp.json();
    assert.ok(analysis.id);
    assert.equal(analysis.experimentId, "exp-123");
    assert.equal(analysis.analysisType, "trajectory");
    assert.equal(analysis.label, "Test trajectory");
    assert.deepEqual(analysis.request, { checkpoint: "/path/to/ckpt" });
    assert.deepEqual(analysis.data, {
      text: "hello",
      trajectory: { gateSignals: [[0.5]] },
    });
    assert.ok(analysis.sizeBytes > 0);
    assert.ok(analysis.createdAt > 0);

    await app.close();
  });

  it("returns 401 without auth", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/experiments/exp-123/analyses",
      payload: samplePayload,
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });

  it("returns 400 for missing analysisType", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/experiments/exp-123/analyses",
      headers: { authorization: AUTH_HEADER },
      payload: { label: "test", request: {}, data: {} },
    });

    assert.equal(resp.statusCode, 400);
    await app.close();
  });

  it("returns 400 for missing data", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/experiments/exp-123/analyses",
      headers: { authorization: AUTH_HEADER },
      payload: { analysisType: "trajectory", label: "test", request: {} },
    });

    assert.equal(resp.statusCode, 400);
    await app.close();
  });
});

// ---------------------------------------------------------------------------
// GET /api/experiments/:expId/analyses
// ---------------------------------------------------------------------------

describe("GET /api/experiments/:expId/analyses", () => {
  it("lists analyses for an experiment", async () => {
    const { app, db } = await buildApp();

    db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "trajectory",
      label: "traj",
      request: {},
      data: {},
    });
    db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "cross-prompt",
      label: "cp",
      request: {},
      data: {},
    });
    db.createAnalysis({
      experimentId: "exp-2",
      analysisType: "sweep",
      label: "sw",
      request: {},
      data: {},
    });

    const resp = await app.inject({
      method: "GET",
      url: "/api/experiments/exp-1/analyses",
    });

    assert.equal(resp.statusCode, 200);
    const { analyses } = resp.json();
    assert.equal(analyses.length, 2);
    // Should not include data blob
    for (const a of analyses) {
      assert.equal("data" in a, false);
      assert.equal("request" in a, false);
    }

    await app.close();
  });

  it("returns empty array for unknown experiment", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "GET",
      url: "/api/experiments/nonexistent/analyses",
    });

    assert.equal(resp.statusCode, 200);
    assert.deepEqual(resp.json().analyses, []);

    await app.close();
  });

  it("filters by analysisType when query param provided", async () => {
    const { app, db } = await buildApp();

    db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "trajectory",
      label: "traj",
      request: {},
      data: {},
    });
    db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "cross-prompt",
      label: "cp",
      request: {},
      data: {},
    });

    const resp = await app.inject({
      method: "GET",
      url: "/api/experiments/exp-1/analyses?type=trajectory",
    });

    assert.equal(resp.statusCode, 200);
    const { analyses } = resp.json();
    assert.equal(analyses.length, 1);
    assert.equal(analyses[0].analysisType, "trajectory");

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// GET /api/analyses/:id
// ---------------------------------------------------------------------------

describe("GET /api/analyses/:id", () => {
  it("returns a full analysis with data", async () => {
    const { app, db } = await buildApp();

    const created = db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "sweep",
      label: "sweep test",
      request: { prompts: ["test"] },
      data: { sweepAnalysis: { configs: [] } },
    });

    const resp = await app.inject({
      method: "GET",
      url: `/api/analyses/${created.id}`,
    });

    assert.equal(resp.statusCode, 200);
    const { analysis } = resp.json();
    assert.equal(analysis.id, created.id);
    assert.deepEqual(analysis.data, { sweepAnalysis: { configs: [] } });
    assert.deepEqual(analysis.request, { prompts: ["test"] });

    await app.close();
  });

  it("returns 404 for unknown id", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "GET",
      url: "/api/analyses/nonexistent",
    });

    assert.equal(resp.statusCode, 404);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// DELETE /api/analyses/:id
// ---------------------------------------------------------------------------

describe("DELETE /api/analyses/:id", () => {
  it("deletes an existing analysis", async () => {
    const { app, db } = await buildApp();

    const created = db.createAnalysis({
      experimentId: "exp-1",
      analysisType: "trajectory",
      label: "to-delete",
      request: {},
      data: {},
    });

    const delResp = await app.inject({
      method: "DELETE",
      url: `/api/analyses/${created.id}`,
      headers: { authorization: AUTH_HEADER },
    });

    assert.equal(delResp.statusCode, 200);
    assert.equal(delResp.json().deleted, true);

    // Confirm it's gone
    const getResp = await app.inject({
      method: "GET",
      url: `/api/analyses/${created.id}`,
    });
    assert.equal(getResp.statusCode, 404);

    await app.close();
  });

  it("returns 404 for non-existent id", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "DELETE",
      url: "/api/analyses/nonexistent",
      headers: { authorization: AUTH_HEADER },
    });

    assert.equal(resp.statusCode, 404);

    await app.close();
  });

  it("returns 401 without auth", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "DELETE",
      url: "/api/analyses/some-id",
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });
});

// ---------------------------------------------------------------------------
// JSON round-trip
// ---------------------------------------------------------------------------

describe("Analysis JSON round-trip", () => {
  it("preserves complex nested data through create → get", async () => {
    const { app } = await buildApp();

    const complexData = {
      batchAnalysis: {
        perPromptSummaries: {
          "Once upon a time": {
            signalStats: {
              modulation: { mean: 0.5, std: 0.1 },
            },
          },
        },
        crossPromptVariance: {
          modulation: { betweenPromptVar: 0.01, withinPromptVar: 0.005 },
        },
      },
      trajectories: {
        "prompt-0": [
          {
            gateSignals: [[0.5], [0.6]],
            effects: [{ temperature: 0.8 }],
            tokenIds: [1, 2],
            tokenTexts: ["Once", " upon"],
          },
        ],
      },
    };

    // Create via API
    const createResp = await app.inject({
      method: "POST",
      url: "/api/experiments/exp-1/analyses",
      headers: { authorization: AUTH_HEADER },
      payload: {
        analysisType: "cross-prompt",
        label: "round-trip test",
        request: { checkpoint: "ckpt" },
        data: complexData,
      },
    });

    assert.equal(createResp.statusCode, 201);
    const { analysis: created } = createResp.json();

    // Retrieve via API
    const getResp = await app.inject({
      method: "GET",
      url: `/api/analyses/${created.id}`,
    });

    assert.equal(getResp.statusCode, 200);
    const { analysis: retrieved } = getResp.json();
    assert.deepEqual(retrieved.data, complexData);

    await app.close();
  });
});
