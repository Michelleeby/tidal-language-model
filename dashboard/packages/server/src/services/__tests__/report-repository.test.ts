import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { Database } from "../database.js";
import { ReportRepository } from "../report-repository.js";
import type { ObjectStore } from "../object-store.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const cleanups: Array<() => Promise<void>> = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "report-repo-test-"));
  cleanups.push(async () => {
    await fsp.rm(dir, { recursive: true, force: true });
  });
  return dir;
}

after(async () => {
  for (const fn of cleanups) await fn();
});

// ---------------------------------------------------------------------------
// Mock ObjectStore
// ---------------------------------------------------------------------------

interface PutCall {
  key: string;
  body: string;
}

function makeMockStore(opts?: {
  putError?: Error;
  getResult?: string;
  listResult?: string[];
}): {
  store: ObjectStore;
  puts: PutCall[];
  deletes: string[];
  deletedPrefixes: string[];
} {
  const puts: PutCall[] = [];
  const deletes: string[] = [];
  const deletedPrefixes: string[] = [];
  const getResult = opts?.getResult;
  const listResult = opts?.listResult ?? [];
  const putError = opts?.putError;

  const mock = {
    isConfigured: () => true,
    putObject: async (key: string, body: string | Buffer | Uint8Array) => {
      if (putError) throw putError;
      puts.push({ key, body: typeof body === "string" ? body : body.toString() });
    },
    putLargeFile: async (_key: string, _filePath: string) => {
      if (putError) throw putError;
    },
    getObject: async (_key: string): Promise<Buffer> => {
      if (getResult !== undefined) return Buffer.from(getResult);
      const err = Object.assign(new Error("NoSuchKey"), { name: "NoSuchKey" });
      throw err;
    },
    headObject: async () => ({ exists: false }),
    listPrefix: async (_prefix: string): Promise<string[]> => listResult,
    deleteObject: async (key: string) => {
      deletes.push(key);
    },
    deletePrefix: async (prefix: string) => {
      deletedPrefixes.push(prefix);
    },
    downloadToFile: async () => {},
  };

  return { store: mock as unknown as ObjectStore, puts, deletes, deletedPrefixes };
}

function makeUnconfiguredStore(): ObjectStore {
  return {
    isConfigured: () => false,
    putObject: async () => {
      throw new Error("ObjectStore not configured");
    },
    putLargeFile: async () => {
      throw new Error("ObjectStore not configured");
    },
    getObject: async () => {
      throw new Error("ObjectStore not configured");
    },
    headObject: async () => ({ exists: false }),
    listPrefix: async () => {
      throw new Error("ObjectStore not configured");
    },
    deleteObject: async () => {
      throw new Error("ObjectStore not configured");
    },
    deletePrefix: async () => {
      throw new Error("ObjectStore not configured");
    },
    downloadToFile: async () => {
      throw new Error("ObjectStore not configured");
    },
  } as unknown as ObjectStore;
}

// ---------------------------------------------------------------------------
// ReportRepository.save() tests
// ---------------------------------------------------------------------------

describe("ReportRepository.save()", () => {
  it("writes to both SQLite and Spaces", async () => {
    const dir = await freshTmpDir();
    const db = new Database(path.join(dir, "test.db"));
    const { store, puts } = makeMockStore();

    const repo = new ReportRepository(db, store);
    const created = repo.create("Test Report");
    const saved = await repo.save(created.id, {
      title: "Updated Title",
      blocks: [{ type: "paragraph" }],
    });

    // SQLite was updated
    assert.equal(saved?.report.title, "Updated Title");
    assert.equal(saved?.spacesWritten, true);
    const fromDb = db.getReport(created.id);
    assert.equal(fromDb?.title, "Updated Title");

    // Spaces was updated: current.json + v{timestamp}.json
    const spacesKeys = puts.map((p) => p.key);
    assert.ok(
      spacesKeys.some((k) => k === `reports/${created.id}/current.json`),
      "Should write current.json",
    );
    assert.ok(
      spacesKeys.some((k) => k.startsWith(`reports/${created.id}/v`) && k.endsWith(".json")),
      "Should write version snapshot",
    );

    db.close();
  });

  it("creates version history (up to 5)", async () => {
    const dir = await freshTmpDir();
    const db = new Database(path.join(dir, "test.db"));

    const versionKeys: string[] = [];
    const store = {
      isConfigured: () => true,
      putObject: async (key: string, _body: string | Buffer | Uint8Array) => {
        if (key.includes("/v") && key.endsWith(".json")) {
          versionKeys.push(key);
        }
      },
      listPrefix: async (_prefix: string): Promise<string[]> => versionKeys.slice(),
      deleteObject: async (key: string) => {
        const idx = versionKeys.indexOf(key);
        if (idx !== -1) versionKeys.splice(idx, 1);
      },
    } as unknown as ObjectStore;

    const repo = new ReportRepository(db, store);
    const created = repo.create("Test Report");

    for (let i = 0; i < 5; i++) {
      await repo.save(created.id, { title: `Save ${i + 1}` });
      await new Promise((r) => setTimeout(r, 2));
    }

    assert.equal(versionKeys.length, 5, "Should have exactly 5 version snapshots");

    db.close();
  });

  it("prunes oldest version when 6th saved", async () => {
    const dir = await freshTmpDir();
    const db = new Database(path.join(dir, "test.db"));

    const versionKeys: string[] = [];
    const store = {
      isConfigured: () => true,
      putObject: async (key: string, _body: string | Buffer | Uint8Array) => {
        if (key.includes("/v") && key.endsWith(".json")) {
          versionKeys.push(key);
        }
      },
      listPrefix: async (_prefix: string): Promise<string[]> => versionKeys.slice(),
      deleteObject: async (key: string) => {
        const idx = versionKeys.indexOf(key);
        if (idx !== -1) versionKeys.splice(idx, 1);
      },
    } as unknown as ObjectStore;

    const repo = new ReportRepository(db, store);
    const created = repo.create("Test Report");

    for (let i = 0; i < 6; i++) {
      await repo.save(created.id, { title: `Save ${i + 1}` });
      await new Promise((r) => setTimeout(r, 2));
    }

    assert.equal(versionKeys.length, 5, "Should prune to 5 versions after 6th save");

    db.close();
  });

  it("writes to SQLite and returns success even when Spaces write fails (503-resilient)", async () => {
    const dir = await freshTmpDir();
    const db = new Database(path.join(dir, "test.db"));
    const { store } = makeMockStore({ putError: new Error("Network error") });

    const repo = new ReportRepository(db, store);
    const created = repo.create("Test Report");

    // Should NOT throw
    const saved = await repo.save(created.id, { title: "Updated", blocks: [] });

    // SQLite update should have worked despite Spaces failure
    assert.equal(saved?.report.title, "Updated");
    assert.equal(saved?.spacesWritten, false);
    assert.ok(saved?.spacesError, "Should include error message");
    const fromDb = db.getReport(created.id);
    assert.equal(fromDb?.title, "Updated");

    db.close();
  });
});

// ---------------------------------------------------------------------------
// ReportRepository.autoSave() tests
// ---------------------------------------------------------------------------

describe("ReportRepository.autoSave()", () => {
  it("writes to SQLite only (no Spaces call)", async () => {
    const dir = await freshTmpDir();
    const db = new Database(path.join(dir, "test.db"));
    const { store, puts } = makeMockStore();

    const repo = new ReportRepository(db, store);
    const created = repo.create("Test Report");
    const saved = repo.autoSave(created.id, { title: "Auto Saved", blocks: [] });

    assert.equal(saved?.title, "Auto Saved");
    assert.equal(puts.length, 0, "autoSave should not write to Spaces");

    db.close();
  });
});

// ---------------------------------------------------------------------------
// ReportRepository.load() tests
// ---------------------------------------------------------------------------

describe("ReportRepository.load()", () => {
  it("returns SQLite data when available", async () => {
    const dir = await freshTmpDir();
    const db = new Database(path.join(dir, "test.db"));
    const { store } = makeMockStore();

    const repo = new ReportRepository(db, store);
    const created = repo.create("Local Report");
    const loaded = await repo.load(created.id);

    assert.equal(loaded?.id, created.id);
    assert.equal(loaded?.title, "Local Report");

    db.close();
  });

  it("falls back to Spaces when not in SQLite", async () => {
    const dir = await freshTmpDir();
    const db = new Database(path.join(dir, "test.db"));

    const spacesReport = {
      id: "remote-report-1",
      userId: null,
      title: "From Spaces",
      blocks: [],
      createdAt: Date.now() - 1000,
      updatedAt: Date.now(),
    };

    const { store } = makeMockStore({ getResult: JSON.stringify(spacesReport) });
    const repo = new ReportRepository(db, store);

    const loaded = await repo.load("remote-report-1");

    assert.equal(loaded?.id, "remote-report-1");
    assert.equal(loaded?.title, "From Spaces");

    db.close();
  });
});

// ---------------------------------------------------------------------------
// ReportRepository.listVersions() tests
// ---------------------------------------------------------------------------

describe("ReportRepository.listVersions()", () => {
  it("returns version keys sorted by timestamp descending", async () => {
    const dir = await freshTmpDir();
    const db = new Database(path.join(dir, "test.db"));

    const reportId = "rpt-123";
    const ts1 = 1700000001000;
    const ts2 = 1700000002000;
    const ts3 = 1700000003000;

    const { store } = makeMockStore({
      listResult: [
        `reports/${reportId}/current.json`,
        `reports/${reportId}/v${ts1}.json`,
        `reports/${reportId}/v${ts3}.json`,
        `reports/${reportId}/v${ts2}.json`,
      ],
    });

    const repo = new ReportRepository(db, store);
    const versions = await repo.listVersions(reportId);

    // Should only include v*.json keys (not current.json)
    assert.equal(versions.length, 3);
    assert.equal(versions[0].timestamp, ts3);
    assert.equal(versions[1].timestamp, ts2);
    assert.equal(versions[2].timestamp, ts1);

    db.close();
  });
});

// ---------------------------------------------------------------------------
// ReportRepository.restoreVersion() tests
// ---------------------------------------------------------------------------

describe("ReportRepository.restoreVersion()", () => {
  it("replaces current with historical version data", async () => {
    const dir = await freshTmpDir();
    const db = new Database(path.join(dir, "test.db"));

    const ts = 1700000001000;
    const historicalData = {
      id: "rpt-restore",
      userId: null,
      title: "Historical Title",
      blocks: [{ type: "heading" }],
      createdAt: ts - 5000,
      updatedAt: ts,
    };

    const puts: Array<{ key: string }> = [];
    const store = {
      isConfigured: () => true,
      getObject: async (key: string): Promise<Buffer> => {
        if (key.endsWith(`v${ts}.json`)) {
          return Buffer.from(JSON.stringify(historicalData));
        }
        throw Object.assign(new Error("NoSuchKey"), { name: "NoSuchKey" });
      },
      putObject: async (key: string, _body: string | Buffer | Uint8Array) => {
        puts.push({ key });
      },
      listPrefix: async (): Promise<string[]> => [],
      deleteObject: async () => {},
    } as unknown as ObjectStore;

    // Create the report in SQLite
    const report = db.createReport("Current Title");
    const repo = new ReportRepository(db, store);

    const restored = await repo.restoreVersion(report.id, ts);

    assert.equal(restored?.title, "Historical Title");
    // current.json in Spaces should be updated
    assert.ok(
      puts.some((p) => p.key.endsWith("current.json")),
      "Should update current.json in Spaces",
    );

    db.close();
  });
});

// ---------------------------------------------------------------------------
// ReportRepository — Spaces not configured (graceful degradation)
// ---------------------------------------------------------------------------

describe("ReportRepository — Spaces not configured", () => {
  it("create, list, autoSave, and load all work without Spaces", async () => {
    const dir = await freshTmpDir();
    const db = new Database(path.join(dir, "test.db"));
    const store = makeUnconfiguredStore();
    const repo = new ReportRepository(db, store);

    const created = repo.create("Report Without Spaces");
    const listed = repo.list();
    assert.ok(listed.some((r) => r.id === created.id));

    const autoSaved = repo.autoSave(created.id, { title: "Auto" });
    assert.equal(autoSaved?.title, "Auto");

    const loaded = await repo.load(created.id);
    assert.equal(loaded?.id, created.id);

    db.close();
  });

  it("save() does not throw when Spaces not configured (SQLite only)", async () => {
    const dir = await freshTmpDir();
    const db = new Database(path.join(dir, "test.db"));
    const store = makeUnconfiguredStore();
    const repo = new ReportRepository(db, store);

    const created = repo.create("Report Without Spaces");
    const saved = await repo.save(created.id, { title: "Saved Without Spaces" });
    assert.equal(saved?.report.title, "Saved Without Spaces");
    assert.equal(saved?.spacesWritten, false);

    db.close();
  });

  it("listVersions() returns empty array when Spaces not configured", async () => {
    const dir = await freshTmpDir();
    const db = new Database(path.join(dir, "test.db"));
    const store = makeUnconfiguredStore();
    const repo = new ReportRepository(db, store);

    const versions = await repo.listVersions("any-report-id");
    assert.deepEqual(versions, []);

    db.close();
  });
});
