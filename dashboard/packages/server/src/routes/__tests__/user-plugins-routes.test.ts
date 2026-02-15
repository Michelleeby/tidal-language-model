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
import type { PluginRegistry } from "../../services/plugin-registry.js";
import { Database } from "../../services/database.js";
import { UserPluginStore } from "../../services/user-plugin-store.js";
import authPlugin from "../../plugins/auth.js";
import userPluginsRoutes from "../user-plugins.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TEST_TOKEN = "test-secret-token";
const JWT_SECRET = "test-jwt-secret-at-least-32-chars-long!";

const cleanups: Array<() => Promise<void>> = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(
    path.join(os.tmpdir(), "user-plugins-route-test-"),
  );
  cleanups.push(async () => {
    await fsp.rm(dir, { recursive: true, force: true });
  });
  return dir;
}

after(async () => {
  for (const fn of cleanups) await fn();
});

/** Create a minimal fake template to copy from. */
async function createFakeTemplate(baseDir: string): Promise<string> {
  const templateDir = path.join(baseDir, "plugins", "tidal");
  await fsp.mkdir(templateDir, { recursive: true });
  await fsp.writeFile(
    path.join(templateDir, "manifest.yaml"),
    "name: tidal\nversion: 1.0\n",
  );
  await fsp.writeFile(
    path.join(templateDir, "Model.py"),
    "class Model:\n    pass\n",
  );
  await fsp.writeFile(path.join(templateDir, "__init__.py"), "");
  const configsDir = path.join(templateDir, "configs");
  await fsp.mkdir(configsDir);
  await fsp.writeFile(
    path.join(configsDir, "base_config.yaml"),
    "EPOCHS: 10\n",
  );
  return templateDir;
}

async function buildApp(): Promise<{
  app: FastifyInstance;
  db: Database;
  tmpDir: string;
}> {
  const tmpDir = await freshTmpDir();
  const db = new Database(path.join(tmpDir, "test.db"));
  const templateDir = await createFakeTemplate(tmpDir);
  const userPluginsDir = path.join(tmpDir, "user-plugins");
  const store = new UserPluginStore(userPluginsDir, templateDir);

  const app = Fastify({ logger: false });
  app.decorate("serverConfig", {
    projectRoot: tmpDir,
    authToken: TEST_TOKEN,
    jwtSecret: JWT_SECRET,
    userPluginsDir,
  } as unknown as ServerConfig);
  app.decorate("db", db);
  app.decorate("userPluginStore", store);

  // Register pluginRegistry stub (for name collision check)
  app.decorate("pluginRegistry", {
    get: (name: string) =>
      name === "tidal" ? ({ name: "tidal" } as never) : undefined,
  } as unknown as PluginRegistry);

  await app.register(cookie);
  await app.register(authPlugin);
  await app.register(userPluginsRoutes);

  cleanups.push(async () => {
    db.close();
  });

  return { app, db, tmpDir };
}

async function createJwt(payload: Record<string, unknown>): Promise<string> {
  const key = new TextEncoder().encode(JWT_SECRET);
  return new SignJWT(payload)
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime("1h")
    .sign(key);
}

async function createUserAndJwt(db: Database, login = "alice") {
  const user = db.upsertUser({
    githubId: Math.floor(Math.random() * 1e9),
    githubLogin: login,
    githubAvatarUrl: null,
  });
  const token = await createJwt({ sub: user.id, githubLogin: login });
  return { user, token };
}

// ---------------------------------------------------------------------------
// Auth enforcement
// ---------------------------------------------------------------------------

describe("User plugins auth enforcement", () => {
  it("returns 401 without auth", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "GET",
      url: "/api/user-plugins",
    });

    assert.equal(resp.statusCode, 401);
    await app.close();
  });

  it("returns 403 with Bearer-only auth (no userId)", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "GET",
      url: "/api/user-plugins",
      headers: { authorization: `Bearer ${TEST_TOKEN}` },
    });

    assert.equal(resp.statusCode, 403);
    await app.close();
  });
});

// ---------------------------------------------------------------------------
// POST /api/user-plugins — create
// ---------------------------------------------------------------------------

describe("POST /api/user-plugins", () => {
  it("creates a plugin from template", async () => {
    const { app, db } = await buildApp();
    const { token } = await createUserAndJwt(db);

    const resp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "my-model", displayName: "My Model" },
    });

    assert.equal(resp.statusCode, 201);
    const body = resp.json();
    assert.ok(body.plugin.id);
    assert.equal(body.plugin.name, "my-model");
    assert.equal(body.plugin.displayName, "My Model");

    await app.close();
  });

  it("rejects invalid plugin names", async () => {
    const { app, db } = await buildApp();
    const { token } = await createUserAndJwt(db);

    const resp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "INVALID!", displayName: "Bad" },
    });

    assert.equal(resp.statusCode, 400);
    await app.close();
  });

  it("rejects names that collide with system plugins", async () => {
    const { app, db } = await buildApp();
    const { token } = await createUserAndJwt(db);

    const resp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "tidal", displayName: "Tidal Clone" },
    });

    assert.equal(resp.statusCode, 409);
    await app.close();
  });

  it("rejects duplicate names for same user", async () => {
    const { app, db } = await buildApp();
    const { token } = await createUserAndJwt(db);

    await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "dup-test", displayName: "First" },
    });

    const resp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "dup-test", displayName: "Second" },
    });

    assert.equal(resp.statusCode, 409);
    await app.close();
  });
});

// ---------------------------------------------------------------------------
// GET /api/user-plugins — list
// ---------------------------------------------------------------------------

describe("GET /api/user-plugins", () => {
  it("returns empty list when no plugins exist", async () => {
    const { app, db } = await buildApp();
    const { token } = await createUserAndJwt(db);

    const resp = await app.inject({
      method: "GET",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
    });

    assert.equal(resp.statusCode, 200);
    assert.deepEqual(resp.json().plugins, []);
    await app.close();
  });

  it("returns only the current user's plugins", async () => {
    const { app, db } = await buildApp();
    const { token: aliceToken } = await createUserAndJwt(db, "alice");
    const { token: bobToken } = await createUserAndJwt(db, "bob");

    await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: aliceToken },
      payload: { name: "alice-model", displayName: "Alice" },
    });

    await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: bobToken },
      payload: { name: "bob-model", displayName: "Bob" },
    });

    const resp = await app.inject({
      method: "GET",
      url: "/api/user-plugins",
      cookies: { tidal_session: aliceToken },
    });

    const { plugins } = resp.json();
    assert.equal(plugins.length, 1);
    assert.equal(plugins[0].name, "alice-model");

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// GET /api/user-plugins/:id — get metadata
// ---------------------------------------------------------------------------

describe("GET /api/user-plugins/:id", () => {
  it("returns plugin metadata", async () => {
    const { app, db } = await buildApp();
    const { token } = await createUserAndJwt(db);

    const createResp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "get-test", displayName: "Get Test" },
    });
    const pluginId = createResp.json().plugin.id;

    const resp = await app.inject({
      method: "GET",
      url: `/api/user-plugins/${pluginId}`,
      cookies: { tidal_session: token },
    });

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.json().plugin.name, "get-test");

    await app.close();
  });

  it("returns 404 for another user's plugin", async () => {
    const { app, db } = await buildApp();
    const { token: aliceToken } = await createUserAndJwt(db, "alice");
    const { token: bobToken } = await createUserAndJwt(db, "bob");

    const createResp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: aliceToken },
      payload: { name: "private-model", displayName: "Private" },
    });
    const pluginId = createResp.json().plugin.id;

    const resp = await app.inject({
      method: "GET",
      url: `/api/user-plugins/${pluginId}`,
      cookies: { tidal_session: bobToken },
    });

    assert.equal(resp.statusCode, 404);
    await app.close();
  });
});

// ---------------------------------------------------------------------------
// PUT /api/user-plugins/:id — update
// ---------------------------------------------------------------------------

describe("PUT /api/user-plugins/:id", () => {
  it("updates display name", async () => {
    const { app, db } = await buildApp();
    const { token } = await createUserAndJwt(db);

    const createResp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "upd-test", displayName: "Old Name" },
    });
    const pluginId = createResp.json().plugin.id;

    const resp = await app.inject({
      method: "PUT",
      url: `/api/user-plugins/${pluginId}`,
      cookies: { tidal_session: token },
      payload: { displayName: "New Name" },
    });

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.json().plugin.displayName, "New Name");

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// DELETE /api/user-plugins/:id — delete
// ---------------------------------------------------------------------------

describe("DELETE /api/user-plugins/:id", () => {
  it("deletes a plugin and its files", async () => {
    const { app, db, tmpDir } = await buildApp();
    const { user, token } = await createUserAndJwt(db);

    const createResp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "del-test", displayName: "Delete Me" },
    });
    const pluginId = createResp.json().plugin.id;

    const resp = await app.inject({
      method: "DELETE",
      url: `/api/user-plugins/${pluginId}`,
      cookies: { tidal_session: token },
    });

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.json().deleted, true);

    // Verify DB record is gone
    assert.equal(db.getUserPlugin(pluginId), null);

    // Verify files are gone
    const pluginDir = path.join(
      tmpDir,
      "user-plugins",
      user.id,
      "del-test",
    );
    await assert.rejects(fsp.access(pluginDir));

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// GET /api/user-plugins/:id/files — file tree
// ---------------------------------------------------------------------------

describe("GET /api/user-plugins/:id/files", () => {
  it("returns the file tree", async () => {
    const { app, db } = await buildApp();
    const { token } = await createUserAndJwt(db);

    const createResp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "tree-model", displayName: "Tree" },
    });
    const pluginId = createResp.json().plugin.id;

    const resp = await app.inject({
      method: "GET",
      url: `/api/user-plugins/${pluginId}/files`,
      cookies: { tidal_session: token },
    });

    assert.equal(resp.statusCode, 200);
    const { files } = resp.json();
    assert.ok(Array.isArray(files));
    const names = files.map((f: { name: string }) => f.name);
    assert.ok(names.includes("manifest.yaml"));

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// GET /api/user-plugins/:id/files/* — read file
// ---------------------------------------------------------------------------

describe("GET /api/user-plugins/:id/files/*", () => {
  it("reads a file by path", async () => {
    const { app, db } = await buildApp();
    const { token } = await createUserAndJwt(db);

    const createResp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "read-model", displayName: "Read" },
    });
    const pluginId = createResp.json().plugin.id;

    const resp = await app.inject({
      method: "GET",
      url: `/api/user-plugins/${pluginId}/files/Model.py`,
      cookies: { tidal_session: token },
    });

    assert.equal(resp.statusCode, 200);
    assert.ok(resp.json().content.includes("class Model"));

    await app.close();
  });

  it("rejects path traversal (framework normalizes URL)", async () => {
    const { app, db } = await buildApp();
    const { token } = await createUserAndJwt(db);

    const createResp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "trav-model", displayName: "Trav" },
    });
    const pluginId = createResp.json().plugin.id;

    // Fastify normalizes ../.. in URLs before route matching,
    // so the request never reaches our handler — returns 404.
    const resp = await app.inject({
      method: "GET",
      url: `/api/user-plugins/${pluginId}/files/../../etc/passwd`,
      cookies: { tidal_session: token },
    });

    // Either 400 (handler caught it) or 404 (framework blocked it) — both are safe
    assert.ok(
      resp.statusCode === 400 || resp.statusCode === 404,
      `Expected 400 or 404, got ${resp.statusCode}`,
    );
    await app.close();
  });
});

// ---------------------------------------------------------------------------
// PUT /api/user-plugins/:id/files/* — write file
// ---------------------------------------------------------------------------

describe("PUT /api/user-plugins/:id/files/*", () => {
  it("saves file content", async () => {
    const { app, db } = await buildApp();
    const { token } = await createUserAndJwt(db);

    const createResp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "write-model", displayName: "Write" },
    });
    const pluginId = createResp.json().plugin.id;

    const writeResp = await app.inject({
      method: "PUT",
      url: `/api/user-plugins/${pluginId}/files/Model.py`,
      cookies: { tidal_session: token },
      payload: { content: "# new content\n" },
    });

    assert.equal(writeResp.statusCode, 200);

    // Verify by reading
    const readResp = await app.inject({
      method: "GET",
      url: `/api/user-plugins/${pluginId}/files/Model.py`,
      cookies: { tidal_session: token },
    });
    assert.equal(readResp.json().content, "# new content\n");

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// POST /api/user-plugins/:id/files/* — create new file
// ---------------------------------------------------------------------------

describe("POST /api/user-plugins/:id/files/*", () => {
  it("creates a new file", async () => {
    const { app, db } = await buildApp();
    const { token } = await createUserAndJwt(db);

    const createResp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "newfile-model", displayName: "NewFile" },
    });
    const pluginId = createResp.json().plugin.id;

    const resp = await app.inject({
      method: "POST",
      url: `/api/user-plugins/${pluginId}/files/NewModule.py`,
      cookies: { tidal_session: token },
      payload: { content: "# new module\n" },
    });

    assert.equal(resp.statusCode, 201);

    // Verify by reading
    const readResp = await app.inject({
      method: "GET",
      url: `/api/user-plugins/${pluginId}/files/NewModule.py`,
      cookies: { tidal_session: token },
    });
    assert.equal(readResp.json().content, "# new module\n");

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// DELETE /api/user-plugins/:id/files/* — delete file
// ---------------------------------------------------------------------------

describe("DELETE /api/user-plugins/:id/files/*", () => {
  it("deletes a file", async () => {
    const { app, db } = await buildApp();
    const { token } = await createUserAndJwt(db);

    const createResp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "delfile-model", displayName: "DelFile" },
    });
    const pluginId = createResp.json().plugin.id;

    const resp = await app.inject({
      method: "DELETE",
      url: `/api/user-plugins/${pluginId}/files/Model.py`,
      cookies: { tidal_session: token },
    });

    assert.equal(resp.statusCode, 200);

    // Verify it's gone
    const readResp = await app.inject({
      method: "GET",
      url: `/api/user-plugins/${pluginId}/files/Model.py`,
      cookies: { tidal_session: token },
    });
    assert.equal(readResp.statusCode, 404);

    await app.close();
  });
});
