import { randomBytes } from "node:crypto";
import { SignJWT, jwtVerify } from "jose";
import type { FastifyInstance } from "fastify";
import type { AuthUser } from "../plugins/auth.js";

export default async function authRoutes(fastify: FastifyInstance) {
  const { githubClientId, githubClientSecret, jwtSecret, publicUrl } =
    fastify.serverConfig;

  const jwtKey = jwtSecret ? new TextEncoder().encode(jwtSecret) : null;

  const oauthConfigured = !!(githubClientId && githubClientSecret && jwtKey);

  // -----------------------------------------------------------------------
  // GET /api/auth/github — redirect to GitHub OAuth
  // -----------------------------------------------------------------------

  fastify.get("/api/auth/github", async (_request, reply) => {
    if (!oauthConfigured) {
      return reply
        .status(501)
        .send({ error: "GitHub OAuth is not configured" });
    }

    const state = randomBytes(20).toString("hex");

    reply.setCookie("oauth_state", state, {
      path: "/",
      httpOnly: true,
      sameSite: "lax",
      maxAge: 600, // 10 minutes
    });

    const params = new URLSearchParams({
      client_id: githubClientId!,
      redirect_uri: `${publicUrl}/api/auth/github/callback`,
      scope: "read:user,public_repo",
      state,
    });

    return reply.redirect(
      `https://github.com/login/oauth/authorize?${params}`,
    );
  });

  // -----------------------------------------------------------------------
  // GET /api/auth/github/callback — exchange code for token
  // -----------------------------------------------------------------------

  fastify.get<{
    Querystring: { code?: string; state?: string };
  }>("/api/auth/github/callback", async (request, reply) => {
    if (!oauthConfigured) {
      return reply
        .status(501)
        .send({ error: "GitHub OAuth is not configured" });
    }

    const { code, state } = request.query;
    const storedState = request.cookies?.["oauth_state"];

    if (!code) {
      return reply.status(400).send({ error: "Missing code parameter" });
    }

    if (!state || !storedState || state !== storedState) {
      return reply.status(400).send({ error: "Invalid or missing state" });
    }

    // Clear state cookie
    reply.clearCookie("oauth_state", { path: "/" });

    // Exchange code for access token
    const tokenResp = await fetch(
      "https://github.com/login/oauth/access_token",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          client_id: githubClientId,
          client_secret: githubClientSecret,
          code,
        }),
      },
    );

    const tokenData = (await tokenResp.json()) as {
      access_token?: string;
      error?: string;
    };

    if (!tokenData.access_token) {
      fastify.log.error("GitHub token exchange failed: %o", tokenData);
      return reply.status(400).send({ error: "Failed to exchange code for token" });
    }

    // Fetch GitHub user info
    const userResp = await fetch("https://api.github.com/user", {
      headers: {
        Authorization: `Bearer ${tokenData.access_token}`,
        Accept: "application/json",
      },
    });

    if (!userResp.ok) {
      fastify.log.error("GitHub user fetch failed: %s", userResp.status);
      return reply.status(400).send({ error: "Failed to fetch GitHub user" });
    }

    const ghUser = (await userResp.json()) as {
      id: number;
      login: string;
      avatar_url: string;
    };

    // Whitelist gate — deny users not on the allowed list
    if (!fastify.db.isUserAllowed(ghUser.login)) {
      fastify.log.warn("GitHub user %s denied: not on whitelist", ghUser.login);
      return reply.redirect("/login?error=not_authorized");
    }

    // Upsert user in database (store token server-side for GitHub API calls)
    const user = fastify.db.upsertUser({
      githubId: ghUser.id,
      githubLogin: ghUser.login,
      githubAvatarUrl: ghUser.avatar_url ?? null,
      githubAccessToken: tokenData.access_token,
    });

    // Create JWT
    const jwt = await new SignJWT({
      sub: user.id,
      githubLogin: user.githubLogin,
    })
      .setProtectedHeader({ alg: "HS256" })
      .setIssuedAt()
      .setExpirationTime("7d")
      .sign(jwtKey!);

    reply.setCookie("tidal_session", jwt, {
      path: "/",
      httpOnly: true,
      sameSite: "lax",
      maxAge: 7 * 24 * 60 * 60, // 7 days
    });

    return reply.redirect("/");
  });

  // -----------------------------------------------------------------------
  // GET /api/auth/me — return current user
  // -----------------------------------------------------------------------

  fastify.get("/api/auth/me", async (request) => {
    if (fastify.serverConfig.devMode) {
      return {
        user: {
          id: "dev",
          githubId: 0,
          githubLogin: "dev",
          githubAvatarUrl: null,
          createdAt: Date.now(),
          lastLoginAt: Date.now(),
        },
      };
    }

    if (!jwtKey) {
      return { user: null };
    }

    const cookieToken = request.cookies?.["tidal_session"] as
      | string
      | undefined;
    if (!cookieToken) {
      return { user: null };
    }

    try {
      const { payload } = await jwtVerify(cookieToken, jwtKey);
      const userId = payload.sub as string;
      const user = fastify.db.getUserById(userId);
      if (!user) return { user: null };
      // Strip server-side-only fields before sending to client
      const { githubAccessToken: _, ...publicUser } = user;
      return { user: publicUser };
    } catch {
      return { user: null };
    }
  });

  // -----------------------------------------------------------------------
  // POST /api/auth/logout — clear session cookie
  // -----------------------------------------------------------------------

  fastify.post("/api/auth/logout", async (_request, reply) => {
    reply.clearCookie("tidal_session", { path: "/" });
    return { loggedOut: true };
  });
}
