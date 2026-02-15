import { describe, it, beforeEach, mock } from "node:test";
import assert from "node:assert/strict";
import { GitHubRepoService } from "../github-repo.js";

// ---------------------------------------------------------------------------
// Mock child_process.execFile
// ---------------------------------------------------------------------------

let execFileCalls: Array<{ file: string; args: string[]; opts: unknown }> = [];
let execFileResult: { stdout: string; stderr: string } = { stdout: "", stderr: "" };
let execFileError: Error | null = null;

const mockExecFile = mock.fn(
  (
    file: string,
    args: string[],
    opts: unknown,
    cb: (err: Error | null, result: { stdout: string; stderr: string }) => void,
  ) => {
    execFileCalls.push({ file, args, opts });
    if (execFileError) {
      cb(execFileError, { stdout: "", stderr: "" });
    } else {
      cb(null, execFileResult);
    }
  },
);

// ---------------------------------------------------------------------------
// Mock global fetch
// ---------------------------------------------------------------------------

let fetchCalls: Array<{ url: string; init?: RequestInit }> = [];
let fetchResponse: { ok: boolean; status: number; json: () => Promise<unknown> } = {
  ok: true,
  status: 200,
  json: async () => ({}),
};

const originalFetch = globalThis.fetch;

function mockFetch(url: string | URL | Request, init?: RequestInit) {
  const urlStr = typeof url === "string" ? url : url.toString();
  fetchCalls.push({ url: urlStr, init });
  return Promise.resolve(fetchResponse as Response);
}

// ---------------------------------------------------------------------------
// Test setup
// ---------------------------------------------------------------------------

function createService() {
  return new GitHubRepoService(mockExecFile as unknown as typeof import("node:child_process").execFile);
}

describe("GitHubRepoService", () => {
  beforeEach(() => {
    execFileCalls = [];
    execFileResult = { stdout: "", stderr: "" };
    execFileError = null;
    fetchCalls = [];
    fetchResponse = {
      ok: true,
      status: 200,
      json: async () => ({}),
    };
    mockExecFile.mock.resetCalls();
    globalThis.fetch = mockFetch as typeof fetch;
  });

  // Restore original fetch after all tests
  // (node:test doesn't have afterAll, but the mock is per-test via beforeEach)

  // -------------------------------------------------------------------------
  // createRepo
  // -------------------------------------------------------------------------

  describe("createRepo()", () => {
    it("calls GitHub API with correct params", async () => {
      fetchResponse = {
        ok: true,
        status: 201,
        json: async () => ({
          html_url: "https://github.com/alice/tidal-plugin-my_model",
          clone_url: "https://github.com/alice/tidal-plugin-my_model.git",
        }),
      };

      const svc = createService();
      const result = await svc.createRepo("gho_token123", "my_model", "My Model plugin");

      assert.equal(fetchCalls.length, 1);
      assert.equal(fetchCalls[0].url, "https://api.github.com/user/repos");
      const body = JSON.parse(fetchCalls[0].init?.body as string);
      assert.equal(body.name, "tidal-plugin-my_model");
      assert.equal(body.description, "My Model plugin");
      assert.equal(body.private, false);
      assert.equal(body.auto_init, false);

      assert.equal(result.htmlUrl, "https://github.com/alice/tidal-plugin-my_model");
      assert.equal(result.cloneUrl, "https://github.com/alice/tidal-plugin-my_model.git");
    });

    it("throws on GitHub API error", async () => {
      fetchResponse = {
        ok: false,
        status: 500,
        json: async () => ({ message: "Internal Server Error" }),
      };

      const svc = createService();
      await assert.rejects(
        svc.createRepo("gho_token123", "my_model", "desc"),
        /Failed to create GitHub repo.*500/,
      );
    });

    it("falls back to existing repo on 422 (name already taken)", async () => {
      let callCount = 0;
      globalThis.fetch = ((url: string | URL | Request, init?: RequestInit) => {
        const urlStr = typeof url === "string" ? url : url.toString();
        fetchCalls.push({ url: urlStr, init });
        callCount++;
        if (callCount === 1) {
          // First call: POST /user/repos → 422
          return Promise.resolve({
            ok: false,
            status: 422,
            json: async () => ({
              message: "Repository creation failed.",
              errors: [{ message: "name already exists on this account" }],
            }),
          } as Response);
        }
        if (callCount === 2) {
          // Second call: GET /user → authenticated user
          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({ login: "alice" }),
          } as Response);
        }
        // Third call: GET /repos/alice/tidal-plugin-my_model → 200
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            html_url: "https://github.com/alice/tidal-plugin-my_model",
            clone_url: "https://github.com/alice/tidal-plugin-my_model.git",
          }),
        } as Response);
      }) as typeof fetch;

      const svc = createService();
      const result = await svc.createRepo("gho_token123", "my_model", "desc");

      assert.equal(result.htmlUrl, "https://github.com/alice/tidal-plugin-my_model");
      assert.equal(result.cloneUrl, "https://github.com/alice/tidal-plugin-my_model.git");
      // Should have made 3 fetch calls: POST create, GET /user, GET /repos/:owner/:repo
      assert.equal(fetchCalls.length, 3);
      assert.ok(fetchCalls[1].url.includes("/user"));
      assert.ok(fetchCalls[2].url.includes("/repos/alice/tidal-plugin-my_model"));
    });
  });

  // -------------------------------------------------------------------------
  // cloneRepo
  // -------------------------------------------------------------------------

  describe("cloneRepo()", () => {
    it("runs git clone to correct directory", async () => {
      const svc = createService();
      await svc.cloneRepo("https://github.com/alice/repo.git", "/tmp/dest");

      assert.equal(execFileCalls.length, 1);
      assert.equal(execFileCalls[0].file, "git");
      assert.deepEqual(execFileCalls[0].args, [
        "clone",
        "https://github.com/alice/repo.git",
        "/tmp/dest",
      ]);
    });
  });

  // -------------------------------------------------------------------------
  // configureGitUser
  // -------------------------------------------------------------------------

  describe("configureGitUser()", () => {
    it("sets git user.name and user.email", async () => {
      const svc = createService();
      await svc.configureGitUser("/tmp/repo", "alice");

      assert.equal(execFileCalls.length, 2);
      assert.deepEqual(execFileCalls[0].args, ["config", "user.name", "alice"]);
      assert.deepEqual(execFileCalls[1].args, ["config", "user.email", "alice@users.noreply.github.com"]);
    });
  });

  // -------------------------------------------------------------------------
  // commitAndPush
  // -------------------------------------------------------------------------

  describe("commitAndPush()", () => {
    it("runs git add, commit, and push with auth remote", async () => {
      const svc = createService();
      await svc.commitAndPush(
        "/tmp/repo",
        "gho_abc",
        "alice",
        "https://github.com/alice/repo.git",
        "Initial commit",
      );

      // Should be: add -A, commit -m, push
      assert.ok(execFileCalls.length >= 3);

      const addCall = execFileCalls.find((c) => c.args[0] === "add");
      assert.ok(addCall);
      assert.deepEqual(addCall!.args, ["add", "-A"]);

      const commitCall = execFileCalls.find((c) => c.args[0] === "commit");
      assert.ok(commitCall);
      assert.ok(commitCall!.args.includes("Initial commit"));

      const pushCall = execFileCalls.find((c) => c.args[0] === "push");
      assert.ok(pushCall);
    });
  });

  // -------------------------------------------------------------------------
  // pull
  // -------------------------------------------------------------------------

  describe("pull()", () => {
    it("runs git pull", async () => {
      const svc = createService();
      await svc.pull("/tmp/repo");

      assert.equal(execFileCalls.length, 1);
      assert.deepEqual(execFileCalls[0].args, ["pull", "origin", "main"]);
    });
  });

  // -------------------------------------------------------------------------
  // getStatus
  // -------------------------------------------------------------------------

  describe("getStatus()", () => {
    it("returns clean status when no changes", async () => {
      execFileResult = { stdout: "", stderr: "" };

      const svc = createService();
      const status = await svc.getStatus("/tmp/repo");

      assert.equal(status.dirty, false);
      assert.deepEqual(status.files, []);
    });

    it("returns dirty status with changed files", async () => {
      execFileResult = {
        stdout: " M Model.py\n?? NewFile.py\n",
        stderr: "",
      };

      const svc = createService();
      const status = await svc.getStatus("/tmp/repo");

      assert.equal(status.dirty, true);
      assert.deepEqual(status.files, ["M Model.py", "?? NewFile.py"]);
    });
  });
});
