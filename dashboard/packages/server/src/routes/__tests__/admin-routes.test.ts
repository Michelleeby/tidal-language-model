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
import authPlugin from "../../plugins/auth.js";
import adminRoutes from "../admin.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const JWT_SECRET = "test-jwt-secret-at-least-32-chars-long!";

const cleanups: Array<() => Promise<void>> = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "admin-route-test-"));
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
    authToken: null,
    jwtSecret: JWT_SECRET,
    githubClientId: null,
    githubClientSecret: null,
    publicUrl: "http://localhost:4400",
  } as unknown as ServerConfig);
  app.decorate("db", db);

  await app.register(cookie);
  await app.register(authPlugin);
  await app.register(adminRoutes);

  cleanups.push(async () => {
    db.close();
  });

  return { app, db };
}

async function createJwt(payload: Record<string, unknown>): Promise<string> {
  const key = new TextEncoder().encode(JWT_SECRET);
  return new SignJWT(payload)
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime("1h")
    .sign(key);
}

async function authedRequest(
  app: FastifyInstance,
  db: Database,
  method: string,
  url: string,
  payload?: unknown,
) {
  const user = db.upsertUser({
    githubId: 1,
    githubLogin: "admin",
    githubAvatarUrl: null,
  });
  const token = await createJwt({ sub: user.id, githubLogin: "admin" });
  return app.inject({
    method: method as "GET" | "POST" | "DELETE",
    url,
    cookies: { tidal_session: token },
    ...(payload ? { payload } : {}),
  });
}

// ---------------------------------------------------------------------------
// Auth enforcement
// ---------------------------------------------------------------------------

describe("Admin routes auth enforcement", () => {
  it("GET /api/admin/allowed-users returns 401 without auth", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "GET",
      url: "/api/admin/allowed-users",
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });

  it("POST /api/admin/allowed-users returns 401 without auth", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "POST",
      url: "/api/admin/allowed-users",
      payload: { githubLogin: "someone" },
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });

  it("DELETE /api/admin/allowed-users/:githubLogin returns 401 without auth", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "DELETE",
      url: "/api/admin/allowed-users/someone",
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });
});

// ---------------------------------------------------------------------------
// GET /api/admin/allowed-users
// ---------------------------------------------------------------------------

describe("GET /api/admin/allowed-users", () => {
  it("returns empty list when no users are whitelisted", async () => {
    const { app, db } = await buildApp();

    const resp = await authedRequest(app, db, "GET", "/api/admin/allowed-users");

    assert.equal(resp.statusCode, 200);
    assert.deepEqual(resp.json().allowedUsers, []);

    await app.close();
  });

  it("returns seeded users", async () => {
    const { app, db } = await buildApp();

    db.addAllowedUser("alice", null);
    db.addAllowedUser("bob", null);

    const resp = await authedRequest(app, db, "GET", "/api/admin/allowed-users");

    assert.equal(resp.statusCode, 200);
    const { allowedUsers } = resp.json();
    assert.equal(allowedUsers.length, 2);
    assert.equal(allowedUsers[0].githubLogin, "alice");
    assert.equal(allowedUsers[1].githubLogin, "bob");

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// POST /api/admin/allowed-users
// ---------------------------------------------------------------------------

describe("POST /api/admin/allowed-users", () => {
  it("creates a new allowed user (201)", async () => {
    const { app, db } = await buildApp();

    const resp = await authedRequest(app, db, "POST", "/api/admin/allowed-users", {
      githubLogin: "newuser",
    });

    assert.equal(resp.statusCode, 201);
    const body = resp.json();
    assert.equal(body.allowedUser.githubLogin, "newuser");
    assert.equal(body.created, true);

    await app.close();
  });

  it("returns 200 for idempotent duplicate", async () => {
    const { app, db } = await buildApp();

    db.addAllowedUser("existing", null);

    const resp = await authedRequest(app, db, "POST", "/api/admin/allowed-users", {
      githubLogin: "existing",
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.allowedUser.githubLogin, "existing");
    assert.equal(body.created, false);

    await app.close();
  });

  it("returns 400 for missing githubLogin", async () => {
    const { app, db } = await buildApp();

    const resp = await authedRequest(app, db, "POST", "/api/admin/allowed-users", {});

    assert.equal(resp.statusCode, 400);

    await app.close();
  });

  it("returns 400 for empty githubLogin", async () => {
    const { app, db } = await buildApp();

    const resp = await authedRequest(app, db, "POST", "/api/admin/allowed-users", {
      githubLogin: "  ",
    });

    assert.equal(resp.statusCode, 400);

    await app.close();
  });

  it("sets addedBy from JWT user", async () => {
    const { app, db } = await buildApp();

    const resp = await authedRequest(app, db, "POST", "/api/admin/allowed-users", {
      githubLogin: "newuser",
    });

    assert.equal(resp.statusCode, 201);
    assert.equal(resp.json().allowedUser.addedBy, "admin");

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// DELETE /api/admin/allowed-users/:githubLogin
// ---------------------------------------------------------------------------

describe("DELETE /api/admin/allowed-users/:githubLogin", () => {
  it("removes an existing user (200)", async () => {
    const { app, db } = await buildApp();

    db.addAllowedUser("toremove", null);

    const resp = await authedRequest(app, db, "DELETE", "/api/admin/allowed-users/toremove");

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.json().removed, true);
    assert.equal(db.isUserAllowed("toremove"), false);

    await app.close();
  });

  it("returns 404 for non-existent user", async () => {
    const { app, db } = await buildApp();

    const resp = await authedRequest(app, db, "DELETE", "/api/admin/allowed-users/nobody");

    assert.equal(resp.statusCode, 404);

    await app.close();
  });

  it("returns 400 for self-removal", async () => {
    const { app, db } = await buildApp();

    db.addAllowedUser("admin", null);

    const resp = await authedRequest(app, db, "DELETE", "/api/admin/allowed-users/admin");

    assert.equal(resp.statusCode, 400);
    assert.match(resp.json().error, /yourself/i);
    // User should still be in the whitelist
    assert.equal(db.isUserAllowed("admin"), true);

    await app.close();
  });
});
