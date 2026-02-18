import { describe, it } from "node:test";
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
import authRoutes from "../auth.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const JWT_SECRET = "test-jwt-secret-at-least-32-chars-long!";
const GITHUB_CLIENT_ID = "test-client-id";
const GITHUB_CLIENT_SECRET = "test-client-secret";

const cleanups: Array<() => Promise<void>> = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "auth-route-test-"));
  cleanups.push(async () => {
    await fsp.rm(dir, { recursive: true, force: true });
  });
  return dir;
}

import { after } from "node:test";
after(async () => {
  for (const fn of cleanups) await fn();
});

async function buildApp(opts?: {
  githubClientId?: string | null;
  githubClientSecret?: string | null;
  jwtSecret?: string | null;
}): Promise<{ app: FastifyInstance; db: Database }> {
  const dir = await freshTmpDir();
  const db = new Database(path.join(dir, "test.db"));

  const app = Fastify({ logger: false });
  app.decorate("serverConfig", {
    authToken: null,
    jwtSecret: opts?.jwtSecret !== undefined ? opts.jwtSecret : JWT_SECRET,
    githubClientId: opts?.githubClientId !== undefined ? opts.githubClientId : GITHUB_CLIENT_ID,
    githubClientSecret: opts?.githubClientSecret !== undefined ? opts.githubClientSecret : GITHUB_CLIENT_SECRET,
    publicUrl: "http://localhost:4400",
  } as unknown as ServerConfig);
  app.decorate("db", db);

  await app.register(cookie);
  await app.register(authPlugin);
  await app.register(authRoutes);

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

// ---------------------------------------------------------------------------
// GET /api/auth/github
// ---------------------------------------------------------------------------

describe("GET /api/auth/github", () => {
  it("returns 501 when GitHub OAuth is not configured", async () => {
    const { app } = await buildApp({
      githubClientId: null,
      githubClientSecret: null,
    });

    const resp = await app.inject({
      method: "GET",
      url: "/api/auth/github",
    });

    assert.equal(resp.statusCode, 501);

    await app.close();
  });

  it("redirects to GitHub when configured", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "GET",
      url: "/api/auth/github",
    });

    assert.equal(resp.statusCode, 302);
    const location = resp.headers.location as string;
    assert.ok(location.startsWith("https://github.com/login/oauth/authorize"));
    assert.ok(location.includes(`client_id=${GITHUB_CLIENT_ID}`));
    assert.ok(location.includes("state="));

    // Should set state cookie
    const cookies = resp.cookies;
    const stateCookie = cookies.find((c: { name: string }) => c.name === "oauth_state");
    assert.ok(stateCookie);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// GET /api/auth/github/callback
// ---------------------------------------------------------------------------

describe("GET /api/auth/github/callback", () => {
  it("returns 400 when state is missing", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "GET",
      url: "/api/auth/github/callback?code=abc",
    });

    assert.equal(resp.statusCode, 400);

    await app.close();
  });

  it("returns 400 when code is missing", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "GET",
      url: "/api/auth/github/callback?state=abc",
      cookies: { oauth_state: "abc" },
    });

    assert.equal(resp.statusCode, 400);

    await app.close();
  });

  it("returns 400 when state does not match cookie", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "GET",
      url: "/api/auth/github/callback?code=abc&state=mismatch",
      cookies: { oauth_state: "expected" },
    });

    assert.equal(resp.statusCode, 400);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// Whitelist gate in callback
// ---------------------------------------------------------------------------

describe("GitHub callback whitelist gate", () => {
  const originalFetch = global.fetch;

  function mockGitHubFetch() {
    global.fetch = (async (url: string | URL | Request) => {
      const urlStr = typeof url === "string" ? url : url.toString();

      if (urlStr.includes("login/oauth/access_token")) {
        return new Response(JSON.stringify({ access_token: "gho_test123" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      if (urlStr.includes("api.github.com/user")) {
        return new Response(
          JSON.stringify({
            id: 42,
            login: "octocat",
            avatar_url: "https://example.com/avatar.png",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }

      return originalFetch(url as RequestInfo, undefined);
    }) as typeof global.fetch;
  }

  function restoreFetch() {
    global.fetch = originalFetch;
  }

  it("non-whitelisted user is redirected to /login?error=not_authorized", async () => {
    const { app, db } = await buildApp();
    mockGitHubFetch();

    try {
      // No users on whitelist — "octocat" is NOT allowed
      const resp = await app.inject({
        method: "GET",
        url: "/api/auth/github/callback?code=testcode&state=teststate",
        cookies: { oauth_state: "teststate" },
      });

      assert.equal(resp.statusCode, 302);
      assert.equal(resp.headers.location, "/login?error=not_authorized");

      // No session cookie should be set
      const sessionCookie = resp.cookies.find(
        (c: { name: string }) => c.name === "tidal_session",
      );
      assert.ok(!sessionCookie || !sessionCookie.value);

      // No user row should exist
      const user = db.getUserByGithubId(42);
      assert.equal(user, null);
    } finally {
      restoreFetch();
      await app.close();
    }
  });

  it("whitelisted user completes login normally", async () => {
    const { app, db } = await buildApp();
    mockGitHubFetch();

    try {
      // Add octocat to whitelist
      db.addAllowedUser("octocat", null);

      const resp = await app.inject({
        method: "GET",
        url: "/api/auth/github/callback?code=testcode&state=teststate",
        cookies: { oauth_state: "teststate" },
      });

      assert.equal(resp.statusCode, 302);
      assert.equal(resp.headers.location, "/");

      // Session cookie should be set
      const sessionCookie = resp.cookies.find(
        (c: { name: string }) => c.name === "tidal_session",
      );
      assert.ok(sessionCookie);
      assert.ok(sessionCookie.value);

      // User row should exist
      const user = db.getUserByGithubId(42);
      assert.ok(user);
      assert.equal(user!.githubLogin, "octocat");
    } finally {
      restoreFetch();
      await app.close();
    }
  });

  it("whitelist check is case-insensitive", async () => {
    const { app, db } = await buildApp();
    mockGitHubFetch();

    try {
      // Add with different case — GitHub returns "octocat"
      db.addAllowedUser("OctoCat", null);

      const resp = await app.inject({
        method: "GET",
        url: "/api/auth/github/callback?code=testcode&state=teststate",
        cookies: { oauth_state: "teststate" },
      });

      assert.equal(resp.statusCode, 302);
      assert.equal(resp.headers.location, "/");
    } finally {
      restoreFetch();
      await app.close();
    }
  });
});

// ---------------------------------------------------------------------------
// GET /api/auth/me
// ---------------------------------------------------------------------------

describe("GET /api/auth/me", () => {
  it("returns null user when not authenticated", async () => {
    const { app } = await buildApp();

    const resp = await app.inject({
      method: "GET",
      url: "/api/auth/me",
    });

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.json().user, null);

    await app.close();
  });

  it("returns user data from valid JWT", async () => {
    const { app, db } = await buildApp();

    // Create a user in the database
    const user = db.upsertUser({
      githubId: 12345,
      githubLogin: "testuser",
      githubAvatarUrl: "https://example.com/avatar.png",
    });

    const token = await createJwt({
      sub: user.id,
      githubLogin: "testuser",
    });

    const resp = await app.inject({
      method: "GET",
      url: "/api/auth/me",
      cookies: { tidal_session: token },
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.user.id, user.id);
    assert.equal(body.user.githubLogin, "testuser");

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// POST /api/auth/logout
// ---------------------------------------------------------------------------

describe("POST /api/auth/logout", () => {
  it("clears the session cookie", async () => {
    const { app } = await buildApp();

    const token = await createJwt({ sub: "user-1", githubLogin: "test" });

    const resp = await app.inject({
      method: "POST",
      url: "/api/auth/logout",
      cookies: { tidal_session: token },
    });

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.json().loggedOut, true);

    // Cookie should be cleared (expired)
    const cookies = resp.cookies;
    const sessionCookie = cookies.find((c: { name: string }) => c.name === "tidal_session");
    assert.ok(sessionCookie);
    // An expired cookie has a date in the past
    assert.ok(
      new Date(sessionCookie.expires!).getTime() <= Date.now(),
    );

    await app.close();
  });
});
