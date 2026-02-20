import { describe, it } from "node:test";
import assert from "node:assert/strict";
import Fastify from "fastify";
import cookie from "@fastify/cookie";
import { SignJWT } from "jose";
import type { FastifyInstance } from "fastify";
import type { ServerConfig } from "../../config.js";
import authPlugin from "../auth.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TEST_TOKEN = "test-bearer-token";
const JWT_SECRET = "test-jwt-secret-at-least-32-chars-long!";

async function buildApp(opts: {
  authToken?: string | null;
  jwtSecret?: string | null;
  devMode?: boolean;
}): Promise<FastifyInstance> {
  const app = Fastify({ logger: false });

  app.decorate("serverConfig", {
    authToken: opts.authToken ?? null,
    jwtSecret: opts.jwtSecret ?? null,
    devMode: opts.devMode ?? false,
  } as unknown as ServerConfig);

  await app.register(cookie);
  await app.register(authPlugin);

  // Test route that requires auth
  app.get("/api/test", { preHandler: [app.verifyAuth] }, async (request) => {
    return { user: request.user };
  });

  return app;
}

async function createJwt(
  secret: string,
  payload: Record<string, unknown>,
): Promise<string> {
  const key = new TextEncoder().encode(secret);
  return new SignJWT(payload)
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime("1h")
    .sign(key);
}

// ---------------------------------------------------------------------------
// Bearer token auth
// ---------------------------------------------------------------------------

describe("Auth plugin — Bearer token", () => {
  it("accepts valid Bearer token", async () => {
    const app = await buildApp({ authToken: TEST_TOKEN });

    const resp = await app.inject({
      method: "GET",
      url: "/api/test",
      headers: { authorization: `Bearer ${TEST_TOKEN}` },
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.user.type, "bearer");

    await app.close();
  });

  it("rejects invalid Bearer token", async () => {
    const app = await buildApp({ authToken: TEST_TOKEN });

    const resp = await app.inject({
      method: "GET",
      url: "/api/test",
      headers: { authorization: "Bearer wrong-token" },
    });

    assert.equal(resp.statusCode, 401);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// JWT cookie auth
// ---------------------------------------------------------------------------

describe("Auth plugin — JWT cookie", () => {
  it("accepts valid JWT in cookie", async () => {
    const app = await buildApp({ jwtSecret: JWT_SECRET });

    const token = await createJwt(JWT_SECRET, {
      sub: "user-123",
      githubLogin: "testuser",
    });

    const resp = await app.inject({
      method: "GET",
      url: "/api/test",
      cookies: { tidal_session: token },
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.user.type, "jwt");
    assert.equal(body.user.userId, "user-123");
    assert.equal(body.user.githubLogin, "testuser");

    await app.close();
  });

  it("rejects expired JWT", async () => {
    const app = await buildApp({ jwtSecret: JWT_SECRET });

    const key = new TextEncoder().encode(JWT_SECRET);
    const token = await new SignJWT({ sub: "user-123", githubLogin: "test" })
      .setProtectedHeader({ alg: "HS256" })
      .setIssuedAt()
      .setExpirationTime("0s") // already expired
      .sign(key);

    // Small delay so it's definitely expired
    await new Promise((r) => setTimeout(r, 1100));

    const resp = await app.inject({
      method: "GET",
      url: "/api/test",
      cookies: { tidal_session: token },
    });

    assert.equal(resp.statusCode, 401);

    await app.close();
  });

  it("rejects JWT signed with wrong secret", async () => {
    const app = await buildApp({ jwtSecret: JWT_SECRET });

    const token = await createJwt("wrong-secret-that-is-also-long!!", {
      sub: "user-123",
      githubLogin: "test",
    });

    const resp = await app.inject({
      method: "GET",
      url: "/api/test",
      cookies: { tidal_session: token },
    });

    assert.equal(resp.statusCode, 401);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// Dual auth (both configured)
// ---------------------------------------------------------------------------

describe("Auth plugin — dual auth", () => {
  it("accepts Bearer when both are configured", async () => {
    const app = await buildApp({ authToken: TEST_TOKEN, jwtSecret: JWT_SECRET });

    const resp = await app.inject({
      method: "GET",
      url: "/api/test",
      headers: { authorization: `Bearer ${TEST_TOKEN}` },
    });

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.json().user.type, "bearer");

    await app.close();
  });

  it("accepts JWT cookie when both are configured", async () => {
    const app = await buildApp({ authToken: TEST_TOKEN, jwtSecret: JWT_SECRET });

    const token = await createJwt(JWT_SECRET, {
      sub: "user-456",
      githubLogin: "dual",
    });

    const resp = await app.inject({
      method: "GET",
      url: "/api/test",
      cookies: { tidal_session: token },
    });

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.json().user.type, "jwt");

    await app.close();
  });

  it("returns 401 with no credentials when both are configured", async () => {
    const app = await buildApp({ authToken: TEST_TOKEN, jwtSecret: JWT_SECRET });

    const resp = await app.inject({
      method: "GET",
      url: "/api/test",
    });

    assert.equal(resp.statusCode, 401);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// No auth configured (both null)
// ---------------------------------------------------------------------------

describe("Auth plugin — no auth configured", () => {
  it("returns 401 when no auth methods are configured", async () => {
    const app = await buildApp({ authToken: null, jwtSecret: null });

    const resp = await app.inject({
      method: "GET",
      url: "/api/test",
    });

    assert.equal(resp.statusCode, 401);

    await app.close();
  });
});

// ---------------------------------------------------------------------------
// Dev mode bypass
// ---------------------------------------------------------------------------

describe("Auth plugin — dev mode", () => {
  it("bypasses auth and sets dev user when devMode is true", async () => {
    const app = await buildApp({ authToken: null, jwtSecret: null, devMode: true });

    const resp = await app.inject({
      method: "GET",
      url: "/api/test",
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.user.type, "jwt");
    assert.equal(body.user.userId, "dev");
    assert.equal(body.user.githubLogin, "dev");

    await app.close();
  });

  it("bypasses auth even when Bearer and JWT are configured", async () => {
    const app = await buildApp({
      authToken: TEST_TOKEN,
      jwtSecret: JWT_SECRET,
      devMode: true,
    });

    // No credentials provided — should still pass
    const resp = await app.inject({
      method: "GET",
      url: "/api/test",
    });

    assert.equal(resp.statusCode, 200);
    assert.equal(resp.json().user.type, "jwt");
    assert.equal(resp.json().user.userId, "dev");

    await app.close();
  });

  it("does not bypass auth when devMode is false", async () => {
    const app = await buildApp({ authToken: null, jwtSecret: null, devMode: false });

    const resp = await app.inject({
      method: "GET",
      url: "/api/test",
    });

    assert.equal(resp.statusCode, 401);

    await app.close();
  });
});
