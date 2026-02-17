import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { ModelSourceBrowser } from "../model-source-browser.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "model-source-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

/** Create a test plugin directory with realistic files. */
async function createTestPluginDir(baseDir: string): Promise<string> {
  const pluginDir = baseDir;

  // Source files
  await fsp.writeFile(path.join(pluginDir, "Main.py"), "# main entry point\n");
  await fsp.writeFile(path.join(pluginDir, "Trainer.py"), "# trainer\n");
  await fsp.writeFile(path.join(pluginDir, "manifest.yaml"), "name: tidal\n");

  // Subdirectory
  const configsDir = path.join(pluginDir, "configs");
  await fsp.mkdir(configsDir, { recursive: true });
  await fsp.writeFile(
    path.join(configsDir, "base_config.yaml"),
    "batch_size: 32\n",
  );

  // Excluded directories
  const pycacheDir = path.join(pluginDir, "__pycache__");
  await fsp.mkdir(pycacheDir, { recursive: true });
  await fsp.writeFile(path.join(pycacheDir, "Main.cpython-311.pyc"), "bytecode");

  const dataCacheDir = path.join(pluginDir, "data_cache");
  await fsp.mkdir(dataCacheDir, { recursive: true });
  await fsp.writeFile(path.join(dataCacheDir, "tokens.bin"), "cached");

  const testsDir = path.join(pluginDir, "tests");
  await fsp.mkdir(testsDir, { recursive: true });
  await fsp.writeFile(path.join(testsDir, "test_model.py"), "# test\n");

  // .pyc file in root (should be excluded)
  await fsp.writeFile(path.join(pluginDir, "compiled.pyc"), "bytecode");

  return pluginDir;
}

// ---------------------------------------------------------------------------
// getFileTree()
// ---------------------------------------------------------------------------

describe("ModelSourceBrowser.getFileTree()", () => {
  it("returns a recursive directory listing", async () => {
    const tmpDir = await freshTmpDir();
    await createTestPluginDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    const tree = await browser.getFileTree();

    // Should have entries at top level
    assert.ok(tree.length > 0, "should return file nodes");

    // Find specific entries
    const mainPy = tree.find((n) => n.name === "Main.py");
    assert.ok(mainPy, "should include Main.py");
    assert.equal(mainPy.type, "file");

    const configs = tree.find((n) => n.name === "configs");
    assert.ok(configs, "should include configs directory");
    assert.equal(configs.type, "directory");
    assert.ok(configs.children);
    assert.ok(configs.children.some((c) => c.name === "base_config.yaml"));
  });

  it("excludes __pycache__ directories", async () => {
    const tmpDir = await freshTmpDir();
    await createTestPluginDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    const tree = await browser.getFileTree();

    const pycache = tree.find((n) => n.name === "__pycache__");
    assert.equal(pycache, undefined, "should not include __pycache__");
  });

  it("excludes .pyc files", async () => {
    const tmpDir = await freshTmpDir();
    await createTestPluginDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    const tree = await browser.getFileTree();

    const pyc = tree.find((n) => n.name === "compiled.pyc");
    assert.equal(pyc, undefined, "should not include .pyc files");
  });

  it("excludes data_cache directories", async () => {
    const tmpDir = await freshTmpDir();
    await createTestPluginDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    const tree = await browser.getFileTree();

    const dataCache = tree.find((n) => n.name === "data_cache");
    assert.equal(dataCache, undefined, "should not include data_cache");
  });

  it("excludes tests/ directories", async () => {
    const tmpDir = await freshTmpDir();
    await createTestPluginDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    const tree = await browser.getFileTree();

    const tests = tree.find((n) => n.name === "tests");
    assert.equal(tests, undefined, "should not include tests/");
  });

  it("sorts directories before files, alphabetical within each", async () => {
    const tmpDir = await freshTmpDir();
    await createTestPluginDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    const tree = await browser.getFileTree();

    // Directories should come before files
    const firstDirIdx = tree.findIndex((n) => n.type === "directory");
    const lastFileIdx = tree.length - 1 - [...tree].reverse().findIndex((n) => n.type === "file");

    if (firstDirIdx !== -1) {
      assert.ok(
        firstDirIdx < lastFileIdx,
        "directories should appear before files",
      );
    }

    // Check alphabetical among files
    const files = tree.filter((n) => n.type === "file");
    for (let i = 1; i < files.length; i++) {
      assert.ok(
        files[i - 1].name.localeCompare(files[i].name) <= 0,
        `files should be alphabetical: ${files[i - 1].name} <= ${files[i].name}`,
      );
    }
  });
});

// ---------------------------------------------------------------------------
// readFile()
// ---------------------------------------------------------------------------

describe("ModelSourceBrowser.readFile()", () => {
  it("returns file content as string", async () => {
    const tmpDir = await freshTmpDir();
    await createTestPluginDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    const content = await browser.readFile("Main.py");
    assert.equal(content, "# main entry point\n");
  });

  it("reads files in subdirectories", async () => {
    const tmpDir = await freshTmpDir();
    await createTestPluginDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    const content = await browser.readFile("configs/base_config.yaml");
    assert.equal(content, "batch_size: 32\n");
  });

  it("rejects path traversal attempts", async () => {
    const tmpDir = await freshTmpDir();
    await createTestPluginDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    await assert.rejects(
      () => browser.readFile("../../../etc/passwd"),
      (err: Error) => {
        assert.ok(err.message.includes("traversal") || err.message.includes("outside"));
        return true;
      },
    );
  });

  it("rejects absolute paths", async () => {
    const tmpDir = await freshTmpDir();
    await createTestPluginDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    await assert.rejects(
      () => browser.readFile("/etc/passwd"),
      (err: Error) => {
        assert.ok(err.message.includes("traversal") || err.message.includes("outside") || err.message.includes("absolute"));
        return true;
      },
    );
  });

  it("throws for non-existent files", async () => {
    const tmpDir = await freshTmpDir();
    await createTestPluginDir(tmpDir);

    const browser = new ModelSourceBrowser(tmpDir);
    await assert.rejects(
      () => browser.readFile("nonexistent.py"),
      (err: Error) => {
        assert.ok(err.message.includes("not found") || err.message.includes("ENOENT"));
        return true;
      },
    );
  });
});
