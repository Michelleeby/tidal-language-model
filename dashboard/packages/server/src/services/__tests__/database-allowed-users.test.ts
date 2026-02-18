import { describe, it, after } from "node:test";
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
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "db-allowed-test-"));
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
// addAllowedUser
// ---------------------------------------------------------------------------

describe("Database.addAllowedUser()", () => {
  it("creates an entry and returns it", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const result = db.addAllowedUser("octocat", "admin-user");

    assert.ok(result);
    assert.ok(result!.id);
    assert.equal(result!.githubLogin, "octocat");
    assert.equal(result!.addedBy, "admin-user");
    assert.ok(result!.createdAt > 0);

    db.close();
  });

  it("returns null for duplicate (INSERT OR IGNORE)", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const first = db.addAllowedUser("octocat", "admin");
    assert.ok(first);

    const second = db.addAllowedUser("octocat", "other-admin");
    assert.equal(second, null);

    db.close();
  });

  it("stores addedBy as null when not provided", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const result = db.addAllowedUser("octocat", null);

    assert.ok(result);
    assert.equal(result!.addedBy, null);

    db.close();
  });
});

// ---------------------------------------------------------------------------
// isUserAllowed
// ---------------------------------------------------------------------------

describe("Database.isUserAllowed()", () => {
  it("returns true for an allowed user", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    db.addAllowedUser("octocat", null);

    assert.equal(db.isUserAllowed("octocat"), true);

    db.close();
  });

  it("returns false for a non-allowed user", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    assert.equal(db.isUserAllowed("stranger"), false);

    db.close();
  });

  it("is case-insensitive", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    db.addAllowedUser("OctoCat", null);

    assert.equal(db.isUserAllowed("octocat"), true);
    assert.equal(db.isUserAllowed("OCTOCAT"), true);
    assert.equal(db.isUserAllowed("OctoCat"), true);

    db.close();
  });
});

// ---------------------------------------------------------------------------
// listAllowedUsers
// ---------------------------------------------------------------------------

describe("Database.listAllowedUsers()", () => {
  it("returns all entries ordered by created_at", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    db.addAllowedUser("alice", "admin");
    // Ensure different timestamps
    await new Promise((r) => setTimeout(r, 15));
    db.addAllowedUser("bob", "admin");

    const users = db.listAllowedUsers();

    assert.equal(users.length, 2);
    assert.equal(users[0].githubLogin, "alice");
    assert.equal(users[1].githubLogin, "bob");

    db.close();
  });

  it("returns empty array when no users exist", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const users = db.listAllowedUsers();
    assert.deepEqual(users, []);

    db.close();
  });
});

// ---------------------------------------------------------------------------
// removeAllowedUser
// ---------------------------------------------------------------------------

describe("Database.removeAllowedUser()", () => {
  it("returns true when removing an existing user", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    db.addAllowedUser("octocat", null);
    const removed = db.removeAllowedUser("octocat");

    assert.equal(removed, true);
    assert.equal(db.isUserAllowed("octocat"), false);

    db.close();
  });

  it("returns false when removing a non-existent user", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    const removed = db.removeAllowedUser("nobody");
    assert.equal(removed, false);

    db.close();
  });

  it("is case-insensitive", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    db.addAllowedUser("OctoCat", null);
    const removed = db.removeAllowedUser("octocat");

    assert.equal(removed, true);
    assert.equal(db.isUserAllowed("OctoCat"), false);

    db.close();
  });
});

// ---------------------------------------------------------------------------
// countAllowedUsers
// ---------------------------------------------------------------------------

describe("Database.countAllowedUsers()", () => {
  it("returns correct count", async () => {
    const dir = await freshTmpDir();
    const db = createDb(dir);

    assert.equal(db.countAllowedUsers(), 0);

    db.addAllowedUser("alice", null);
    assert.equal(db.countAllowedUsers(), 1);

    db.addAllowedUser("bob", null);
    assert.equal(db.countAllowedUsers(), 2);

    db.removeAllowedUser("alice");
    assert.equal(db.countAllowedUsers(), 1);

    db.close();
  });
});
