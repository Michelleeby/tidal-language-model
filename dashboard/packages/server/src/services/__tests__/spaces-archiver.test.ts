import { describe, it, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { SpacesArchiver } from "../spaces-archiver.js";
import { ObjectStore } from "../object-store.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const cleanups: Array<() => Promise<void>> = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "spaces-archiver-test-"));
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

interface PutCall { key: string; size: number }
interface HeadCall { key: string; result: boolean }

function makeMockStore(opts?: {
  headResult?: boolean;
  putError?: Error;
}): {
  store: ObjectStore;
  puts: PutCall[];
  deletes: string[];
  deletedPrefixes: string[];
} {
  const puts: PutCall[] = [];
  const deletes: string[] = [];
  const deletedPrefixes: string[] = [];
  const headResult = opts?.headResult ?? true;
  const putError = opts?.putError;

  const mock = {
    isConfigured: () => true,
    putObject: async (key: string, body: string | Buffer | Uint8Array) => {
      if (putError) throw putError;
      puts.push({ key, size: typeof body === "string" ? body.length : body.byteLength });
    },
    putLargeFile: async (key: string, _filePath: string) => {
      if (putError) throw putError;
      puts.push({ key, size: 0 });
    },
    headObject: async (_key: string) => ({
      exists: headResult,
      sizeBytes: headResult ? 1024 : undefined,
    }),
    deleteObject: async (key: string) => {
      deletes.push(key);
    },
    deletePrefix: async (prefix: string) => {
      deletedPrefixes.push(prefix);
    },
    listPrefix: async () => [] as string[],
    getObject: async () => Buffer.from(""),
    downloadToFile: async () => {},
  };

  return { store: mock as unknown as ObjectStore, puts, deletes, deletedPrefixes };
}

async function createExperiment(
  experimentsDir: string,
  expId: string,
  files: string[],
): Promise<void> {
  const expDir = path.join(experimentsDir, expId);
  await fsp.mkdir(expDir, { recursive: true });
  for (const file of files) {
    const filePath = path.join(expDir, file);
    await fsp.mkdir(path.dirname(filePath), { recursive: true });
    await fsp.writeFile(filePath, `content of ${file}`);
  }
}

async function readManifest(experimentsDir: string, expId: string) {
  const manifestPath = path.join(experimentsDir, expId, "_archive_manifest.json");
  const raw = await fsp.readFile(manifestPath, "utf-8");
  return JSON.parse(raw);
}

// ---------------------------------------------------------------------------
// SpacesArchiver tests
// ---------------------------------------------------------------------------

describe("SpacesArchiver.archiveExperiment()", () => {
  it("uploads all .pth files to correct Spaces prefix", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const { store, puts } = makeMockStore();

    await createExperiment(experimentsDir, "exp-1", [
      "checkpoint_foundational_epoch_1.pth",
      "checkpoint_foundational_epoch_2.pth",
      "metadata.json",
    ]);

    const archiver = new SpacesArchiver(store, experimentsDir);
    await archiver.archiveExperiment("exp-1");

    // Should upload both .pth files
    const pthUploads = puts.filter((p) => p.key.endsWith(".pth"));
    assert.equal(pthUploads.length, 2);
    assert.ok(pthUploads.some((p) => p.key.includes("checkpoint_foundational_epoch_1.pth")));
    assert.ok(pthUploads.some((p) => p.key.includes("checkpoint_foundational_epoch_2.pth")));
  });

  it("deletes .pth files from disk after successful upload", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const { store } = makeMockStore({ headResult: true });

    await createExperiment(experimentsDir, "exp-2", [
      "model.pth",
      "metadata.json",
    ]);

    const archiver = new SpacesArchiver(store, experimentsDir);
    await archiver.archiveExperiment("exp-2");

    // .pth file should be gone
    const pthExists = await fsp
      .access(path.join(experimentsDir, "exp-2", "model.pth"))
      .then(() => true)
      .catch(() => false);
    assert.equal(pthExists, false);
  });

  it("retains metadata.json, config.yaml, dashboard_metrics/", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const { store } = makeMockStore({ headResult: true });

    await createExperiment(experimentsDir, "exp-3", [
      "model.pth",
      "metadata.json",
      "config.yaml",
      "dashboard_metrics/status.json",
    ]);

    const archiver = new SpacesArchiver(store, experimentsDir);
    await archiver.archiveExperiment("exp-3");

    // These files should still exist
    for (const file of ["metadata.json", "config.yaml", "dashboard_metrics/status.json"]) {
      const exists = await fsp
        .access(path.join(experimentsDir, "exp-3", file))
        .then(() => true)
        .catch(() => false);
      assert.equal(exists, true, `${file} should be retained`);
    }
  });

  it("writes _archive_manifest.json with state:uploading then state:complete", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const { store } = makeMockStore({ headResult: true });

    await createExperiment(experimentsDir, "exp-4", ["model.pth"]);

    const archiver = new SpacesArchiver(store, experimentsDir);
    await archiver.archiveExperiment("exp-4");

    const manifest = await readManifest(experimentsDir, "exp-4");
    assert.equal(manifest.state, "complete");
    assert.ok(manifest.archivedAt > 0);
    assert.ok(manifest.spacesPrefix.includes("exp-4"));
    assert.ok(Array.isArray(manifest.archivedFiles));
  });

  it("writes state:failed and preserves local files on upload failure", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const uploadError = new Error("Network error");
    const { store } = makeMockStore({ putError: uploadError });

    await createExperiment(experimentsDir, "exp-5", ["model.pth"]);

    const archiver = new SpacesArchiver(store, experimentsDir);
    await archiver.archiveExperiment("exp-5");

    // Manifest should show failed
    const manifest = await readManifest(experimentsDir, "exp-5");
    assert.equal(manifest.state, "failed");

    // Local file should still exist
    const pthExists = await fsp
      .access(path.join(experimentsDir, "exp-5", "model.pth"))
      .then(() => true)
      .catch(() => false);
    assert.equal(pthExists, true, ".pth should be preserved on failure");
  });

  it("is idempotent (second call on state:complete is a no-op)", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const { store, puts } = makeMockStore({ headResult: true });

    await createExperiment(experimentsDir, "exp-6", ["model.pth"]);

    const archiver = new SpacesArchiver(store, experimentsDir);
    await archiver.archiveExperiment("exp-6");

    const putsAfterFirst = puts.length;

    // Second call — should be no-op
    await archiver.archiveExperiment("exp-6");
    assert.equal(puts.length, putsAfterFirst, "No new uploads on second call");
  });

  it("does not delete local files if HEAD verification fails", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    // headResult:false means the file wasn't actually stored
    const { store } = makeMockStore({ headResult: false });

    await createExperiment(experimentsDir, "exp-7", ["model.pth"]);

    const archiver = new SpacesArchiver(store, experimentsDir);
    await archiver.archiveExperiment("exp-7");

    // Local file should be preserved since HEAD says it doesn't exist in Spaces
    const pthExists = await fsp
      .access(path.join(experimentsDir, "exp-7", "model.pth"))
      .then(() => true)
      .catch(() => false);
    assert.equal(pthExists, true, ".pth should be preserved when HEAD verification fails");

    // Manifest should be failed
    const manifest = await readManifest(experimentsDir, "exp-7");
    assert.equal(manifest.state, "failed");
  });

  it("retrieveFile() downloads from Spaces and writes to disk", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");

    const fileContent = Buffer.from("fake checkpoint data");
    const mock = {
      isConfigured: () => true,
      headObject: async () => ({ exists: true, sizeBytes: fileContent.length }),
      getObject: async (_key: string) => fileContent,
      downloadToFile: async (key: string, destPath: string) => {
        await fsp.writeFile(destPath, fileContent);
      },
    };

    // Create experiment directory (simulating a completed+archived experiment)
    const expDir = path.join(experimentsDir, "exp-8");
    await fsp.mkdir(expDir, { recursive: true });

    // Write a complete manifest
    await fsp.writeFile(
      path.join(expDir, "_archive_manifest.json"),
      JSON.stringify({
        state: "complete",
        archivedAt: Date.now(),
        spacesPrefix: "experiments/exp-8/",
        archivedFiles: ["model.pth"],
      }),
    );

    const archiver = new SpacesArchiver(mock as unknown as ObjectStore, experimentsDir);
    await archiver.retrieveFile("exp-8", "model.pth");

    // File should now exist on disk
    const exists = await fsp
      .access(path.join(expDir, "model.pth"))
      .then(() => true)
      .catch(() => false);
    assert.equal(exists, true, "Retrieved file should exist on disk");
  });
});

// ---------------------------------------------------------------------------
// SpacesArchiver.isArchived() / getManifest()
// ---------------------------------------------------------------------------

describe("SpacesArchiver.isArchived()", () => {
  it("returns false for experiment with no manifest", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const { store } = makeMockStore();
    await createExperiment(experimentsDir, "exp-check", ["metadata.json"]);

    const archiver = new SpacesArchiver(store, experimentsDir);
    assert.equal(await archiver.isArchived("exp-check"), false);
  });

  it("returns true for experiment with state:complete manifest", async () => {
    const dir = await freshTmpDir();
    const experimentsDir = path.join(dir, "experiments");
    const { store } = makeMockStore();
    await createExperiment(experimentsDir, "exp-done", ["metadata.json"]);

    await fsp.writeFile(
      path.join(experimentsDir, "exp-done", "_archive_manifest.json"),
      JSON.stringify({ state: "complete", archivedAt: Date.now(), spacesPrefix: "experiments/exp-done/" }),
    );

    const archiver = new SpacesArchiver(store, experimentsDir);
    assert.equal(await archiver.isArchived("exp-done"), true);
  });
});
