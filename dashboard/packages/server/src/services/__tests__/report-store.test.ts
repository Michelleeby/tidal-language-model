import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { ReportStore } from "../report-store.js";

// ---------------------------------------------------------------------------
// Temp directory management
// ---------------------------------------------------------------------------

const cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "report-store-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// ReportStore.create()
// ---------------------------------------------------------------------------

describe("ReportStore.create()", () => {
  it("creates a report with a generated id and default title", async () => {
    const dir = await freshTmpDir();
    const store = new ReportStore(dir);

    const report = await store.create();

    assert.ok(report.id, "should have an id");
    assert.equal(report.title, "Untitled Report");
    assert.deepEqual(report.blocks, []);
    assert.ok(report.createdAt > 0);
    assert.ok(report.updatedAt > 0);
  });

  it("creates a report with a custom title", async () => {
    const dir = await freshTmpDir();
    const store = new ReportStore(dir);

    const report = await store.create("My Report");

    assert.equal(report.title, "My Report");
  });

  it("persists the report as a JSON file", async () => {
    const dir = await freshTmpDir();
    const store = new ReportStore(dir);

    const report = await store.create();
    const filePath = path.join(dir, `${report.id}.json`);

    const raw = await fsp.readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw);
    assert.equal(parsed.id, report.id);
    assert.equal(parsed.title, "Untitled Report");
  });
});

// ---------------------------------------------------------------------------
// ReportStore.get()
// ---------------------------------------------------------------------------

describe("ReportStore.get()", () => {
  it("returns a previously created report", async () => {
    const dir = await freshTmpDir();
    const store = new ReportStore(dir);

    const created = await store.create("Test");
    const fetched = await store.get(created.id);

    assert.ok(fetched);
    assert.equal(fetched!.id, created.id);
    assert.equal(fetched!.title, "Test");
  });

  it("returns null for a non-existent id", async () => {
    const dir = await freshTmpDir();
    const store = new ReportStore(dir);

    const result = await store.get("nonexistent");

    assert.equal(result, null);
  });
});

// ---------------------------------------------------------------------------
// ReportStore.list()
// ---------------------------------------------------------------------------

describe("ReportStore.list()", () => {
  it("returns empty array when no reports exist", async () => {
    const dir = await freshTmpDir();
    const store = new ReportStore(dir);

    const reports = await store.list();

    assert.deepEqual(reports, []);
  });

  it("returns summaries sorted by updatedAt descending", async () => {
    const dir = await freshTmpDir();
    const store = new ReportStore(dir);

    const r1 = await store.create("First");
    // Ensure different timestamps
    await new Promise((r) => setTimeout(r, 10));
    const r2 = await store.create("Second");

    const reports = await store.list();

    assert.equal(reports.length, 2);
    // Most recently updated first
    assert.equal(reports[0].id, r2.id);
    assert.equal(reports[1].id, r1.id);
    // Summaries should not include blocks
    assert.equal((reports[0] as any).blocks, undefined);
  });
});

// ---------------------------------------------------------------------------
// ReportStore.update()
// ---------------------------------------------------------------------------

describe("ReportStore.update()", () => {
  it("updates the title", async () => {
    const dir = await freshTmpDir();
    const store = new ReportStore(dir);

    const created = await store.create();
    const updated = await store.update(created.id, { title: "New Title" });

    assert.ok(updated);
    assert.equal(updated!.title, "New Title");
    assert.ok(updated!.updatedAt >= created.updatedAt);
  });

  it("updates the blocks", async () => {
    const dir = await freshTmpDir();
    const store = new ReportStore(dir);

    const created = await store.create();
    const blocks = [{ type: "paragraph", content: "hello" }];
    const updated = await store.update(created.id, { blocks });

    assert.ok(updated);
    assert.deepEqual(updated!.blocks, blocks);
  });

  it("returns null for a non-existent id", async () => {
    const dir = await freshTmpDir();
    const store = new ReportStore(dir);

    const result = await store.update("nonexistent", { title: "X" });

    assert.equal(result, null);
  });

  it("persists updates to disk", async () => {
    const dir = await freshTmpDir();
    const store = new ReportStore(dir);

    const created = await store.create();
    await store.update(created.id, { title: "Persisted" });

    const raw = await fsp.readFile(path.join(dir, `${created.id}.json`), "utf-8");
    const parsed = JSON.parse(raw);
    assert.equal(parsed.title, "Persisted");
  });
});

// ---------------------------------------------------------------------------
// ReportStore.delete()
// ---------------------------------------------------------------------------

describe("ReportStore.delete()", () => {
  it("deletes an existing report", async () => {
    const dir = await freshTmpDir();
    const store = new ReportStore(dir);

    const created = await store.create();
    const deleted = await store.delete(created.id);

    assert.equal(deleted, true);

    const fetched = await store.get(created.id);
    assert.equal(fetched, null);
  });

  it("returns false for a non-existent id", async () => {
    const dir = await freshTmpDir();
    const store = new ReportStore(dir);

    const deleted = await store.delete("nonexistent");

    assert.equal(deleted, false);
  });

  it("removes the JSON file from disk", async () => {
    const dir = await freshTmpDir();
    const store = new ReportStore(dir);

    const created = await store.create();
    await store.delete(created.id);

    const files = await fsp.readdir(dir);
    assert.equal(files.length, 0);
  });
});
