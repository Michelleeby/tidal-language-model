import { describe, it, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { Database } from "../database.js";

// ---------------------------------------------------------------------------
// Temp directory management
// ---------------------------------------------------------------------------

const cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "db-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

function createDb(dir: string): Database {
  return new Database(path.join(dir, "test.db"));
}

// ---------------------------------------------------------------------------
// Schema creation
// ---------------------------------------------------------------------------

describe("Database schema", () => {
  it("creates tables on construction", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    // Should not throw — tables exist
    const users = db.listUsers();
    assert.deepEqual(users, []);

    const reports = db.listReports();
    assert.deepEqual(reports, []);

    db.close();
  });

  it("opens an existing database without errors", async () => {
    const dir = await freshTmpDir();
    const dbPath = path.join(dir, "test.db");

    const db1 = new Database(dbPath);
    db1.close();

    // Re-open same file
    const db2 = new Database(dbPath);
    const users = db2.listUsers();
    assert.deepEqual(users, []);
    db2.close();
  });
});

// ---------------------------------------------------------------------------
// User operations
// ---------------------------------------------------------------------------

describe("Database.upsertUser()", () => {
  it("inserts a new user", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const user = db.upsertUser({
      githubId: 12345,
      githubLogin: "testuser",
      githubAvatarUrl: "https://avatars.githubusercontent.com/u/12345",
    });

    assert.ok(user.id);
    assert.equal(user.githubId, 12345);
    assert.equal(user.githubLogin, "testuser");
    assert.equal(user.githubAvatarUrl, "https://avatars.githubusercontent.com/u/12345");
    assert.ok(user.createdAt > 0);
    assert.ok(user.lastLoginAt > 0);

    db.close();
  });

  it("updates last_login_at on conflict", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const first = db.upsertUser({
      githubId: 12345,
      githubLogin: "testuser",
      githubAvatarUrl: null,
    });

    // Upsert again
    const second = db.upsertUser({
      githubId: 12345,
      githubLogin: "testuser-updated",
      githubAvatarUrl: "https://example.com/avatar.png",
    });

    // Same internal id
    assert.equal(second.id, first.id);
    // Updated fields
    assert.equal(second.githubLogin, "testuser-updated");
    assert.equal(second.githubAvatarUrl, "https://example.com/avatar.png");
    assert.ok(second.lastLoginAt >= first.lastLoginAt);

    db.close();
  });
});

describe("Database.upsertUser() with githubAccessToken", () => {
  it("stores and retrieves githubAccessToken", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const user = db.upsertUser({
      githubId: 99999,
      githubLogin: "tokenuser",
      githubAvatarUrl: null,
      githubAccessToken: "gho_abc123",
    });

    assert.ok(user.id);
    const retrieved = db.getUserById(user.id);
    assert.ok(retrieved);
    assert.equal(retrieved!.githubAccessToken, "gho_abc123");

    db.close();
  });

  it("updates githubAccessToken on conflict", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    db.upsertUser({
      githubId: 88888,
      githubLogin: "tokenuser2",
      githubAvatarUrl: null,
      githubAccessToken: "old_token",
    });

    db.upsertUser({
      githubId: 88888,
      githubLogin: "tokenuser2",
      githubAvatarUrl: null,
      githubAccessToken: "new_token",
    });

    const found = db.getUserByGithubId(88888);
    assert.ok(found);
    assert.equal(found!.githubAccessToken, "new_token");

    db.close();
  });

  it("stores null when no token provided", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const user = db.upsertUser({
      githubId: 77777,
      githubLogin: "notoken",
      githubAvatarUrl: null,
    });

    const retrieved = db.getUserById(user.id);
    assert.ok(retrieved);
    assert.equal(retrieved!.githubAccessToken, null);

    db.close();
  });
});

describe("Database.getUserById()", () => {
  it("returns a user by id", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const created = db.upsertUser({
      githubId: 111,
      githubLogin: "alice",
      githubAvatarUrl: null,
    });

    const found = db.getUserById(created.id);
    assert.ok(found);
    assert.equal(found!.githubLogin, "alice");

    db.close();
  });

  it("returns null for non-existent id", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const found = db.getUserById("nonexistent");
    assert.equal(found, null);

    db.close();
  });
});

describe("Database.getUserByGithubId()", () => {
  it("returns a user by github id", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    db.upsertUser({ githubId: 222, githubLogin: "bob", githubAvatarUrl: null });

    const found = db.getUserByGithubId(222);
    assert.ok(found);
    assert.equal(found!.githubLogin, "bob");

    db.close();
  });

  it("returns null for non-existent github id", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const found = db.getUserByGithubId(999);
    assert.equal(found, null);

    db.close();
  });
});

// ---------------------------------------------------------------------------
// getUserByGithubLogin
// ---------------------------------------------------------------------------

describe("Database.getUserByGithubLogin()", () => {
  it("finds a user by github login", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    db.upsertUser({ githubId: 555, githubLogin: "octocat", githubAvatarUrl: null });

    const found = db.getUserByGithubLogin("octocat");
    assert.ok(found);
    assert.equal(found!.githubLogin, "octocat");
    assert.equal(found!.githubId, 555);

    db.close();
  });

  it("returns null for unknown login", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const found = db.getUserByGithubLogin("nonexistent");
    assert.equal(found, null);

    db.close();
  });
});

// ---------------------------------------------------------------------------
// Report operations
// ---------------------------------------------------------------------------

describe("Database.createReport()", () => {
  it("creates a report with default title and null userId", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const report = db.createReport();

    assert.ok(report.id);
    assert.equal(report.userId, null);
    assert.equal(report.title, "Untitled Report");
    assert.deepEqual(report.blocks, []);
    assert.ok(report.createdAt > 0);
    assert.ok(report.updatedAt > 0);

    db.close();
  });

  it("creates a report with custom title and userId", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const user = db.upsertUser({
      githubId: 1,
      githubLogin: "test",
      githubAvatarUrl: null,
    });

    const report = db.createReport("My Report", user.id);

    assert.equal(report.title, "My Report");
    assert.equal(report.userId, user.id);

    db.close();
  });
});

describe("Database.getReport()", () => {
  it("returns a report by id", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const created = db.createReport("Test");
    const found = db.getReport(created.id);

    assert.ok(found);
    assert.equal(found!.id, created.id);
    assert.equal(found!.title, "Test");

    db.close();
  });

  it("returns null for non-existent id", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const found = db.getReport("nonexistent");
    assert.equal(found, null);

    db.close();
  });
});

describe("Database.listReports()", () => {
  it("returns all reports sorted by updatedAt descending", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    db.createReport("First");
    // Ensure a different updatedAt timestamp
    await new Promise((r) => setTimeout(r, 15));
    db.createReport("Second");

    const reports = db.listReports();

    assert.equal(reports.length, 2);
    // Most recently updated first
    assert.equal(reports[0].title, "Second");
    assert.equal(reports[1].title, "First");

    db.close();
  });
});

describe("Database.listReportsByUser()", () => {
  it("returns only reports for a specific user", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const alice = db.upsertUser({ githubId: 1, githubLogin: "alice", githubAvatarUrl: null });
    const bob = db.upsertUser({ githubId: 2, githubLogin: "bob", githubAvatarUrl: null });

    db.createReport("Alice Report", alice.id);
    db.createReport("Bob Report", bob.id);
    db.createReport("No owner");

    const aliceReports = db.listReportsByUser(alice.id);
    assert.equal(aliceReports.length, 1);
    assert.equal(aliceReports[0].title, "Alice Report");

    db.close();
  });
});

describe("Database.updateReport()", () => {
  it("updates title", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const created = db.createReport();
    const updated = db.updateReport(created.id, { title: "New Title" });

    assert.ok(updated);
    assert.equal(updated!.title, "New Title");
    assert.ok(updated!.updatedAt >= created.updatedAt);

    db.close();
  });

  it("updates blocks", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const created = db.createReport();
    const blocks = [{ type: "paragraph", content: "hello" }];
    const updated = db.updateReport(created.id, { blocks });

    assert.ok(updated);
    assert.deepEqual(updated!.blocks, blocks);

    db.close();
  });

  it("returns null for non-existent id", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const result = db.updateReport("nonexistent", { title: "X" });
    assert.equal(result, null);

    db.close();
  });
});

describe("Database.deleteReport()", () => {
  it("deletes an existing report", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const created = db.createReport();
    const deleted = db.deleteReport(created.id);
    assert.equal(deleted, true);

    const found = db.getReport(created.id);
    assert.equal(found, null);

    db.close();
  });

  it("returns false for non-existent id", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const deleted = db.deleteReport("nonexistent");
    assert.equal(deleted, false);

    db.close();
  });
});

// ---------------------------------------------------------------------------
// importLegacyReport
// ---------------------------------------------------------------------------

describe("Database.importLegacyReport()", () => {
  it("imports a report with INSERT OR IGNORE", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const now = Date.now();
    const imported = db.importLegacyReport({
      id: "legacy-1",
      title: "Legacy",
      blocks: [{ type: "paragraph" }],
      createdAt: now,
      updatedAt: now,
    });

    assert.equal(imported, true);

    const found = db.getReport("legacy-1");
    assert.ok(found);
    assert.equal(found!.title, "Legacy");
    assert.equal(found!.userId, null);

    db.close();
  });

  it("ignores duplicate imports", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const now = Date.now();
    const report = { id: "dup-1", title: "First", blocks: [], createdAt: now, updatedAt: now };

    const first = db.importLegacyReport(report);
    assert.equal(first, true);

    const second = db.importLegacyReport({ ...report, title: "Second" });
    assert.equal(second, false);

    // Original still has first title
    const found = db.getReport("dup-1");
    assert.equal(found!.title, "First");

    db.close();
  });
});
