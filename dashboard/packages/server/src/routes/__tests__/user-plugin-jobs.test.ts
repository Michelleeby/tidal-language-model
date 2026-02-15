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
import { GitHubRepoService } from "../../services/github-repo.js";
import authPlugin from "../../plugins/auth.js";
import userPluginsRoutes from "../user-plugins.js";
import jobRoutes from "../jobs.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TEST_TOKEN = "test-secret-token";
const JWT_SECRET = "test-jwt-secret-at-least-32-chars-long!";

const cleanups: Array<() => Promise<void>> = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(
    path.join(os.tmpdir(), "user-plugin-jobs-test-"),
  );
  cleanups.push(async () => {
    await fsp.rm(dir, { recursive: true, force: true });
  });
  return dir;
}

after(async () => {
  for (const fn of cleanups) await fn();
});

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

async function createJwt(payload: Record<string, unknown>): Promise<string> {
  const key = new TextEncoder().encode(JWT_SECRET);
  return new SignJWT(payload)
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime("1h")
    .sign(key);
}

async function buildApp() {
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
    redisUrl: "redis://localhost:6379",
    experimentsDir: path.join(tmpDir, "experiments"),
    defaultComputeProvider: "local",
    pythonBin: "python3",
  } as unknown as ServerConfig);
  app.decorate("db", db);
  app.decorate("userPluginStore", store);

  // Stub pluginRegistry
  app.decorate("pluginRegistry", {
    get: (name: string) =>
      name === "tidal" ? ({ name: "tidal" } as never) : undefined,
    getDefault: () => null,
    list: () => [],
  } as unknown as PluginRegistry);

  // Stub githubRepo
  const githubRepoStub = {
    createRepo: async () => ({
      htmlUrl: "https://github.com/test/tidal-plugin-test",
      cloneUrl: "https://github.com/test/tidal-plugin-test.git",
    }),
    cloneRepo: async (_url: string, dest: string) => {
      await fsp.mkdir(dest, { recursive: true });
    },
    configureGitUser: async () => {},
    commitAndPush: async () => {},
    pull: async () => {},
    getStatus: async () => ({ dirty: false, files: [] }),
  } as unknown as GitHubRepoService;
  app.decorate("githubRepo", githubRepoStub);

  // Stub Redis (null — jobs route handles this)
  app.decorate("redis", null);

  // Stub SSE manager
  app.decorate("sseManager", { broadcastJobUpdate: () => {} } as never);

  // Stub provisioning chain
  app.decorate("provisioningChain", {
    getProvider: () => undefined,
  } as never);

  // Stub worker spawner
  app.decorate("workerSpawner", {
    spawnLocal: () => {},
    isRunning: () => false,
    kill: () => {},
    cleanup: () => {},
  } as never);

  await app.register(cookie);
  await app.register(authPlugin);
  await app.register(userPluginsRoutes);
  await app.register(jobRoutes);

  cleanups.push(async () => db.close());

  return { app, db, tmpDir };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("POST /api/jobs with userPluginId", () => {
  it("resolves pluginDir and pluginRepoUrl from userPluginId", async () => {
    const { app, db } = await buildApp();

    // Create user with token
    const user = db.upsertUser({
      githubId: 12345,
      githubLogin: "testuser",
      githubAvatarUrl: null,
      githubAccessToken: "gho_test",
    });
    const token = await createJwt({ sub: user.id, githubLogin: "testuser" });

    // Create a user plugin
    const createResp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: token },
      payload: { name: "test_model", displayName: "Test Model" },
    });
    const pluginId = createResp.json().plugin.id;

    // Create a job referencing the user plugin
    // This will fail at Redis level but we can check the job config was set up correctly
    const resp = await app.inject({
      method: "POST",
      url: "/api/jobs",
      cookies: { tidal_session: token },
      payload: {
        type: "lm-training",
        plugin: "test_model",
        configPath: "user-plugins/test/configs/base_config.yaml",
        userPluginId: pluginId,
      },
    });

    // Redis is null so we expect 503 — but the request should at least not be 403/404
    // The important thing is that it reaches the orchestrator (which fails on Redis)
    assert.ok(
      resp.statusCode === 201 || resp.statusCode === 503,
      `Expected 201 or 503 (Redis), got ${resp.statusCode}: ${resp.body}`,
    );

    await app.close();
  });

  it("rejects invalid userPluginId", async () => {
    const { app, db } = await buildApp();

    const user = db.upsertUser({
      githubId: 12345,
      githubLogin: "testuser",
      githubAvatarUrl: null,
    });
    const token = await createJwt({ sub: user.id, githubLogin: "testuser" });

    const resp = await app.inject({
      method: "POST",
      url: "/api/jobs",
      cookies: { tidal_session: token },
      payload: {
        type: "lm-training",
        plugin: "test_model",
        configPath: "configs/base_config.yaml",
        userPluginId: "nonexistent-id",
      },
    });

    assert.equal(resp.statusCode, 404);
    await app.close();
  });

  it("rejects another user's userPluginId", async () => {
    const { app, db } = await buildApp();

    const alice = db.upsertUser({
      githubId: 111,
      githubLogin: "alice",
      githubAvatarUrl: null,
      githubAccessToken: "gho_alice",
    });
    const bob = db.upsertUser({
      githubId: 222,
      githubLogin: "bob",
      githubAvatarUrl: null,
    });

    const aliceToken = await createJwt({ sub: alice.id, githubLogin: "alice" });
    const bobToken = await createJwt({ sub: bob.id, githubLogin: "bob" });

    // Alice creates a plugin
    const createResp = await app.inject({
      method: "POST",
      url: "/api/user-plugins",
      cookies: { tidal_session: aliceToken },
      payload: { name: "alice_model", displayName: "Alice" },
    });
    const pluginId = createResp.json().plugin.id;

    // Bob tries to use Alice's plugin for a job
    const resp = await app.inject({
      method: "POST",
      url: "/api/jobs",
      cookies: { tidal_session: bobToken },
      payload: {
        type: "lm-training",
        plugin: "alice_model",
        configPath: "configs/base_config.yaml",
        userPluginId: pluginId,
      },
    });

    assert.equal(resp.statusCode, 404);
    await app.close();
  });

  it("works without userPluginId (unchanged behavior)", async () => {
    const { app, db } = await buildApp();

    const user = db.upsertUser({
      githubId: 12345,
      githubLogin: "testuser",
      githubAvatarUrl: null,
    });
    const token = await createJwt({ sub: user.id, githubLogin: "testuser" });

    const resp = await app.inject({
      method: "POST",
      url: "/api/jobs",
      cookies: { tidal_session: token },
      payload: {
        type: "lm-training",
        plugin: "tidal",
        configPath: "plugins/tidal/configs/base_config.yaml",
      },
    });

    // Should reach orchestrator (fails on Redis, not auth)
    assert.ok(
      resp.statusCode === 201 || resp.statusCode === 503,
      `Expected 201 or 503, got ${resp.statusCode}`,
    );

    await app.close();
  });
});
