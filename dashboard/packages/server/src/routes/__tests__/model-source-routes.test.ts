import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { ModelSourceBrowser } from "../../services/model-source-browser.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "model-route-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

async function createTestSourceDir(baseDir: string): Promise<string> {
  await fsp.writeFile(path.join(baseDir, "Main.py"), "# main entry\n");
  await fsp.writeFile(path.join(baseDir, "Trainer.py"), "class Trainer:\n    pass\n");
  const configs = path.join(baseDir, "configs");
  await fsp.mkdir(configs, { recursive: true });
  await fsp.writeFile(path.join(configs, "base_config.yaml"), "lr: 0.001\n");
  // Excluded
  const pycache = path.join(baseDir, "__pycache__");
  await fsp.mkdir(pycache, { recursive: true });
  await fsp.writeFile(path.join(pycache, "Main.cpython-311.pyc"), "bytecode");
  return baseDir;
}

// ---------------------------------------------------------------------------
// GET /api/model/files — file tree (simulated route handler logic)
// ---------------------------------------------------------------------------

describe("GET /api/model/files (via ModelSourceBrowser)", () => {
  it("returns file tree", async () => {
    const tmpDir = await freshTmpDir();
    await createTestSourceDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    const files = await browser.getFileTree();

    assert.ok(files.length > 0);
    const mainPy = files.find((f) => f.name === "Main.py");
    assert.ok(mainPy);
    assert.equal(mainPy.type, "file");

    const configs = files.find((f) => f.name === "configs");
    assert.ok(configs);
    assert.equal(configs.type, "directory");
    assert.ok(configs.children?.some((c) => c.name === "base_config.yaml"));
  });

  it("excludes __pycache__", async () => {
    const tmpDir = await freshTmpDir();
    await createTestSourceDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    const files = await browser.getFileTree();

    const pycache = files.find((f) => f.name === "__pycache__");
    assert.equal(pycache, undefined);
  });
});

// ---------------------------------------------------------------------------
// GET /api/model/files/:path — file content
// ---------------------------------------------------------------------------

describe("GET /api/model/files/:path (via ModelSourceBrowser)", () => {
  it("returns file content for valid path", async () => {
    const tmpDir = await freshTmpDir();
    await createTestSourceDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    const content = await browser.readFile("Main.py");
    assert.equal(content, "# main entry\n");
  });

  it("returns nested file content", async () => {
    const tmpDir = await freshTmpDir();
    await createTestSourceDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    const content = await browser.readFile("configs/base_config.yaml");
    assert.equal(content, "lr: 0.001\n");
  });

  it("rejects path traversal attempts (../../etc/passwd)", async () => {
    const tmpDir = await freshTmpDir();
    await createTestSourceDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    await assert.rejects(
      () => browser.readFile("../../etc/passwd"),
      (err: Error) => {
        assert.ok(err.message.includes("traversal") || err.message.includes("outside"));
        return true;
      },
    );
  });

  it("returns error for non-existent file", async () => {
    const tmpDir = await freshTmpDir();
    await createTestSourceDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    await assert.rejects(
      () => browser.readFile("nonexistent.py"),
      (err: Error) => {
        assert.ok(err.message.includes("not found"));
        return true;
      },
    );
  });
});

// ---------------------------------------------------------------------------
// No POST/PUT/DELETE — read-only verification
// ---------------------------------------------------------------------------

describe("Model source routes are read-only", () => {
  it("ModelSourceBrowser has no write methods", () => {
    const browser = new ModelSourceBrowser("/tmp");
    const methods = Object.getOwnPropertyNames(Object.getPrototypeOf(browser));
    const writeMethods = methods.filter((m) =>
      ["writeFile", "saveFile", "deleteFile", "createFile"].includes(m),
    );
    assert.equal(writeMethods.length, 0, "should have no write methods");
  });
});
