import { describe, it, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { UserPluginStore } from "../user-plugin-store.js";

// ---------------------------------------------------------------------------
// Temp directory management
// ---------------------------------------------------------------------------

const cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "plugin-store-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

/**
 * Create a minimal fake template directory to act as the system plugin.
 * Includes a manifest.yaml, a .py file, a configs/ dir, __pycache__/, and a .pyc.
 */
async function createFakeTemplate(baseDir: string): Promise<string> {
  const templateDir = path.join(baseDir, "plugins", "tidal");
  await fsp.mkdir(templateDir, { recursive: true });
  await fsp.writeFile(
    path.join(templateDir, "manifest.yaml"),
    "name: tidal\nversion: 1.0\n",
  );
  await fsp.writeFile(
    path.join(templateDir, "Model.py"),
    "class Model:\n    pass\n",
  );
  await fsp.writeFile(path.join(templateDir, "__init__.py"), "");

  // configs/
  const configsDir = path.join(templateDir, "configs");
  await fsp.mkdir(configsDir);
  await fsp.writeFile(
    path.join(configsDir, "base_config.yaml"),
    "EPOCHS: 10\n",
  );

  // __pycache__/ (should be excluded)
  const cacheDir = path.join(templateDir, "__pycache__");
  await fsp.mkdir(cacheDir);
  await fsp.writeFile(path.join(cacheDir, "Model.cpython-312.pyc"), "fake");

  // data_cache/ (should be excluded)
  const dataCache = path.join(templateDir, "data_cache");
  await fsp.mkdir(dataCache);
  await fsp.writeFile(path.join(dataCache, "stuff.bin"), "data");

  // .env (should be excluded)
  await fsp.writeFile(path.join(templateDir, ".env"), "SECRET=123");

  return baseDir;
}

function createStore(
  userPluginsDir: string,
  templateDir: string,
): UserPluginStore {
  return new UserPluginStore(userPluginsDir, templateDir);
}

// ---------------------------------------------------------------------------
// createFromTemplate
// ---------------------------------------------------------------------------

describe("UserPluginStore.createFromTemplate()", () => {
  it("copies template files into user plugin directory", async () => {
    const baseDir = await freshTmpDir();
    await createFakeTemplate(baseDir);
    const userPluginsDir = path.join(baseDir, "user-plugins");
    const templateDir = path.join(baseDir, "plugins", "tidal");
    const store = createStore(userPluginsDir, templateDir);

    await store.createFromTemplate("user1", "my-model");

    // Check copied files
    const manifest = await fsp.readFile(
      path.join(userPluginsDir, "user1", "my-model", "manifest.yaml"),
      "utf-8",
    );
    assert.ok(manifest.includes("name:"));

    const model = await fsp.readFile(
      path.join(userPluginsDir, "user1", "my-model", "Model.py"),
      "utf-8",
    );
    assert.ok(model.includes("class Model"));

    const config = await fsp.readFile(
      path.join(userPluginsDir, "user1", "my-model", "configs", "base_config.yaml"),
      "utf-8",
    );
    assert.ok(config.includes("EPOCHS"));
  });

  it("excludes __pycache__, data_cache, .env, .pyc", async () => {
    const baseDir = await freshTmpDir();
    await createFakeTemplate(baseDir);
    const userPluginsDir = path.join(baseDir, "user-plugins");
    const templateDir = path.join(baseDir, "plugins", "tidal");
    const store = createStore(userPluginsDir, templateDir);

    await store.createFromTemplate("user1", "clean-model");

    const pluginDir = path.join(userPluginsDir, "user1", "clean-model");

    // __pycache__ should not exist
    await assert.rejects(fsp.access(path.join(pluginDir, "__pycache__")));
    // data_cache should not exist
    await assert.rejects(fsp.access(path.join(pluginDir, "data_cache")));
    // .env should not exist
    await assert.rejects(fsp.access(path.join(pluginDir, ".env")));
  });
});

// ---------------------------------------------------------------------------
// getFileTree
// ---------------------------------------------------------------------------

describe("UserPluginStore.getFileTree()", () => {
  it("returns recursive file tree", async () => {
    const baseDir = await freshTmpDir();
    await createFakeTemplate(baseDir);
    const userPluginsDir = path.join(baseDir, "user-plugins");
    const templateDir = path.join(baseDir, "plugins", "tidal");
    const store = createStore(userPluginsDir, templateDir);

    await store.createFromTemplate("user1", "tree-test");

    const tree = await store.getFileTree("user1", "tree-test");

    // Should have top-level files
    const names = tree.map((n) => n.name);
    assert.ok(names.includes("manifest.yaml"));
    assert.ok(names.includes("Model.py"));

    // Should have configs/ directory with children
    const configsNode = tree.find((n) => n.name === "configs");
    assert.ok(configsNode);
    assert.equal(configsNode!.type, "directory");
    assert.ok(configsNode!.children!.length > 0);
    assert.ok(
      configsNode!.children!.some((c) => c.name === "base_config.yaml"),
    );
  });
});

// ---------------------------------------------------------------------------
// readFile
// ---------------------------------------------------------------------------

describe("UserPluginStore.readFile()", () => {
  it("reads a file by relative path", async () => {
    const baseDir = await freshTmpDir();
    await createFakeTemplate(baseDir);
    const userPluginsDir = path.join(baseDir, "user-plugins");
    const templateDir = path.join(baseDir, "plugins", "tidal");
    const store = createStore(userPluginsDir, templateDir);

    await store.createFromTemplate("user1", "read-test");

    const content = await store.readFile("user1", "read-test", "Model.py");
    assert.ok(content.includes("class Model"));
  });

  it("reads nested files", async () => {
    const baseDir = await freshTmpDir();
    await createFakeTemplate(baseDir);
    const userPluginsDir = path.join(baseDir, "user-plugins");
    const templateDir = path.join(baseDir, "plugins", "tidal");
    const store = createStore(userPluginsDir, templateDir);

    await store.createFromTemplate("user1", "read-test2");

    const content = await store.readFile(
      "user1",
      "read-test2",
      "configs/base_config.yaml",
    );
    assert.ok(content.includes("EPOCHS"));
  });

  it("rejects path traversal", async () => {
    const baseDir = await freshTmpDir();
    await createFakeTemplate(baseDir);
    const userPluginsDir = path.join(baseDir, "user-plugins");
    const templateDir = path.join(baseDir, "plugins", "tidal");
    const store = createStore(userPluginsDir, templateDir);

    await store.createFromTemplate("user1", "traversal-test");

    await assert.rejects(
      store.readFile("user1", "traversal-test", "../../etc/passwd"),
      /path/i,
    );
  });
});

// ---------------------------------------------------------------------------
// writeFile
// ---------------------------------------------------------------------------

describe("UserPluginStore.writeFile()", () => {
  it("saves content to a file", async () => {
    const baseDir = await freshTmpDir();
    await createFakeTemplate(baseDir);
    const userPluginsDir = path.join(baseDir, "user-plugins");
    const templateDir = path.join(baseDir, "plugins", "tidal");
    const store = createStore(userPluginsDir, templateDir);

    await store.createFromTemplate("user1", "write-test");
    await store.writeFile("user1", "write-test", "Model.py", "# updated\n");

    const content = await store.readFile("user1", "write-test", "Model.py");
    assert.equal(content, "# updated\n");
  });

  it("rejects disallowed extensions", async () => {
    const baseDir = await freshTmpDir();
    await createFakeTemplate(baseDir);
    const userPluginsDir = path.join(baseDir, "user-plugins");
    const templateDir = path.join(baseDir, "plugins", "tidal");
    const store = createStore(userPluginsDir, templateDir);

    await store.createFromTemplate("user1", "ext-test");

    await assert.rejects(
      store.writeFile("user1", "ext-test", "evil.exe", "data"),
      /extension/i,
    );
  });

  it("rejects path traversal on write", async () => {
    const baseDir = await freshTmpDir();
    await createFakeTemplate(baseDir);
    const userPluginsDir = path.join(baseDir, "user-plugins");
    const templateDir = path.join(baseDir, "plugins", "tidal");
    const store = createStore(userPluginsDir, templateDir);

    await store.createFromTemplate("user1", "traversal-write");

    await assert.rejects(
      store.writeFile("user1", "traversal-write", "../../../etc/evil.py", "bad"),
      /path/i,
    );
  });

  it("rejects files over 1MB", async () => {
    const baseDir = await freshTmpDir();
    await createFakeTemplate(baseDir);
    const userPluginsDir = path.join(baseDir, "user-plugins");
    const templateDir = path.join(baseDir, "plugins", "tidal");
    const store = createStore(userPluginsDir, templateDir);

    await store.createFromTemplate("user1", "size-test");

    const bigContent = "x".repeat(1024 * 1024 + 1);
    await assert.rejects(
      store.writeFile("user1", "size-test", "big.py", bigContent),
      /size/i,
    );
  });
});

// ---------------------------------------------------------------------------
// deleteFile
// ---------------------------------------------------------------------------

describe("UserPluginStore.deleteFile()", () => {
  it("deletes a file", async () => {
    const baseDir = await freshTmpDir();
    await createFakeTemplate(baseDir);
    const userPluginsDir = path.join(baseDir, "user-plugins");
    const templateDir = path.join(baseDir, "plugins", "tidal");
    const store = createStore(userPluginsDir, templateDir);

    await store.createFromTemplate("user1", "del-test");
    await store.deleteFile("user1", "del-test", "Model.py");

    await assert.rejects(store.readFile("user1", "del-test", "Model.py"));
  });

  it("refuses to delete manifest.yaml", async () => {
    const baseDir = await freshTmpDir();
    await createFakeTemplate(baseDir);
    const userPluginsDir = path.join(baseDir, "user-plugins");
    const templateDir = path.join(baseDir, "plugins", "tidal");
    const store = createStore(userPluginsDir, templateDir);

    await store.createFromTemplate("user1", "manifest-test");

    await assert.rejects(
      store.deleteFile("user1", "manifest-test", "manifest.yaml"),
      /manifest/i,
    );
  });
});

// ---------------------------------------------------------------------------
// deletePlugin
// ---------------------------------------------------------------------------

describe("UserPluginStore.deletePlugin()", () => {
  it("removes the entire plugin directory", async () => {
    const baseDir = await freshTmpDir();
    await createFakeTemplate(baseDir);
    const userPluginsDir = path.join(baseDir, "user-plugins");
    const templateDir = path.join(baseDir, "plugins", "tidal");
    const store = createStore(userPluginsDir, templateDir);

    await store.createFromTemplate("user1", "rm-test");
    await store.deletePlugin("user1", "rm-test");

    const pluginDir = path.join(userPluginsDir, "user1", "rm-test");
    await assert.rejects(fsp.access(pluginDir));
  });
});
