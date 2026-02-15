import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { Database } from "../database.js";
import { migrateLegacyReports } from "../report-migration.js";

// ---------------------------------------------------------------------------
// Temp directory management
// ---------------------------------------------------------------------------

const cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "migration-test-"));
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

/** Minimal logger satisfying the log interface. */
const nullLog = {
  info: () => {},
  warn: () => {},
  error: () => {},
};

// ---------------------------------------------------------------------------
// migrateLegacyReports()
// ---------------------------------------------------------------------------

describe("migrateLegacyReports()", () => {
  it("imports JSON report files into the database", async () => {
    const dir = await freshTmpDir();
    const reportsDir = path.join(dir, "reports");
    await fsp.mkdir(reportsDir);

    const now = Date.now();
    await fsp.writeFile(
      path.join(reportsDir, "r1.json"),
      JSON.stringify({
        id: "r1",
        title: "Report One",
        blocks: [{ type: "paragraph" }],
        createdAt: now,
        updatedAt: now,
      }),
    );
    await fsp.writeFile(
      path.join(reportsDir, "r2.json"),
      JSON.stringify({
        id: "r2",
        title: "Report Two",
        blocks: [],
        createdAt: now,
        updatedAt: now,
      }),
    );

    const db = createDb(dir);
    const result = await migrateLegacyReports(reportsDir, db, nullLog);

    assert.equal(result.imported, 2);
    assert.equal(result.skipped, 0);
    assert.equal(result.errors, 0);

    const r1 = db.getReport("r1");
    assert.ok(r1);
    assert.equal(r1!.title, "Report One");
    assert.equal(r1!.userId, null);

    db.close();
  });

  it("skips duplicate reports", async () => {
    const dir = await freshTmpDir();
    const reportsDir = path.join(dir, "reports");
    await fsp.mkdir(reportsDir);

    const now = Date.now();
    await fsp.writeFile(
      path.join(reportsDir, "r1.json"),
      JSON.stringify({ id: "r1", title: "X", blocks: [], createdAt: now, updatedAt: now }),
    );

    const db = createDb(dir);

    // Import once
    await migrateLegacyReports(reportsDir, db, nullLog);
    // Import again
    const result = await migrateLegacyReports(reportsDir, db, nullLog);

    assert.equal(result.imported, 0);
    assert.equal(result.skipped, 1);

    db.close();
  });

  it("handles empty directory", async () => {
    const dir = await freshTmpDir();
    const reportsDir = path.join(dir, "reports");
    await fsp.mkdir(reportsDir);

    const db = createDb(dir);
    const result = await migrateLegacyReports(reportsDir, db, nullLog);

    assert.equal(result.imported, 0);
    assert.equal(result.skipped, 0);
    assert.equal(result.errors, 0);

    db.close();
  });

  it("handles missing directory", async () => {
    const dir = await freshTmpDir();
    const reportsDir = path.join(dir, "does-not-exist");

    const db = createDb(dir);
    const result = await migrateLegacyReports(reportsDir, db, nullLog);

    assert.equal(result.imported, 0);
    assert.equal(result.skipped, 0);
    assert.equal(result.errors, 0);

    db.close();
  });

  it("counts malformed JSON as errors", async () => {
    const dir = await freshTmpDir();
    const reportsDir = path.join(dir, "reports");
    await fsp.mkdir(reportsDir);

    await fsp.writeFile(path.join(reportsDir, "bad.json"), "not-json{{{");

    const db = createDb(dir);
    const result = await migrateLegacyReports(reportsDir, db, nullLog);

    assert.equal(result.imported, 0);
    assert.equal(result.errors, 1);

    db.close();
  });

  it("skips non-JSON files", async () => {
    const dir = await freshTmpDir();
    const reportsDir = path.join(dir, "reports");
    await fsp.mkdir(reportsDir);

    await fsp.writeFile(path.join(reportsDir, "readme.txt"), "hello");

    const now = Date.now();
    await fsp.writeFile(
      path.join(reportsDir, "r1.json"),
      JSON.stringify({ id: "r1", title: "X", blocks: [], createdAt: now, updatedAt: now }),
    );

    const db = createDb(dir);
    const result = await migrateLegacyReports(reportsDir, db, nullLog);

    assert.equal(result.imported, 1);
    assert.equal(result.errors, 0);

    db.close();
  });
});
