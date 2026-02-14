import { describe, it, after, before } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "config-route-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

/**
 * Simulate the route handler logic for reading config files.
 * This mirrors the implementation we'll build in configs.ts.
 */
const SAFE_FILENAME = /^[a-zA-Z0-9_-]+\.(yaml|yml)$/;

function isValidConfigFilename(filename: string): boolean {
  return SAFE_FILENAME.test(filename);
}

async function readConfigFile(
  pluginsDir: string,
  pluginName: string,
  filename: string,
): Promise<{ status: number; body: Record<string, unknown> }> {
  if (!isValidConfigFilename(filename)) {
    return { status: 400, body: { error: "Invalid filename" } };
  }

  const configDir = path.join(pluginsDir, pluginName, "configs");
  const filePath = path.join(configDir, filename);

  // Prevent path traversal: resolved path must be inside configDir
  const resolved = path.resolve(filePath);
  if (!resolved.startsWith(path.resolve(configDir) + path.sep) && resolved !== path.resolve(configDir)) {
    return { status: 400, body: { error: "Invalid filename" } };
  }

  try {
    const content = await fsp.readFile(resolved, "utf-8");
    return { status: 200, body: { filename, content } };
  } catch {
    return { status: 404, body: { error: "Config file not found" } };
  }
}

async function listConfigFiles(
  pluginsDir: string,
  pluginName: string,
): Promise<{ status: number; body: Record<string, unknown> }> {
  const configDir = path.join(pluginsDir, pluginName, "configs");
  try {
    const entries = await fsp.readdir(configDir);
    const yamlFiles = entries.filter((f) => /\.(yaml|yml)$/.test(f)).sort();
    return { status: 200, body: { plugin: pluginName, files: yamlFiles } };
  } catch {
    return { status: 404, body: { error: "Plugin config directory not found" } };
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("Config filename validation", () => {
  it("accepts valid YAML filenames", () => {
    assert.ok(isValidConfigFilename("base_config.yaml"));
    assert.ok(isValidConfigFilename("rl_config.yaml"));
    assert.ok(isValidConfigFilename("my-config.yml"));
    assert.ok(isValidConfigFilename("Config123.yaml"));
  });

  it("rejects path traversal attempts", () => {
    assert.ok(!isValidConfigFilename("../../../etc/passwd"));
    assert.ok(!isValidConfigFilename("..%2F..%2Fetc%2Fpasswd"));
    assert.ok(!isValidConfigFilename("foo/../bar.yaml"));
  });

  it("rejects non-YAML extensions", () => {
    assert.ok(!isValidConfigFilename("config.json"));
    assert.ok(!isValidConfigFilename("config.txt"));
    assert.ok(!isValidConfigFilename("script.py"));
  });

  it("rejects filenames with path separators", () => {
    assert.ok(!isValidConfigFilename("subdir/config.yaml"));
    assert.ok(!isValidConfigFilename("/etc/config.yaml"));
  });

  it("rejects empty or whitespace filenames", () => {
    assert.ok(!isValidConfigFilename(""));
    assert.ok(!isValidConfigFilename(" "));
    assert.ok(!isValidConfigFilename(".yaml"));
  });
});

describe("GET /api/plugins/:name/configs/:filename (via file read)", () => {
  it("returns config file content for valid filename", async () => {
    const tmpDir = await freshTmpDir();
    const configDir = path.join(tmpDir, "tidal", "configs");
    await fsp.mkdir(configDir, { recursive: true });
    await fsp.writeFile(
      path.join(configDir, "base_config.yaml"),
      "VOCAB_SIZE: 50257\nEMBED_DIM: 256\n",
    );

    const result = await readConfigFile(tmpDir, "tidal", "base_config.yaml");
    assert.equal(result.status, 200);
    assert.equal(result.body.filename, "base_config.yaml");
    assert.ok((result.body.content as string).includes("VOCAB_SIZE: 50257"));
  });

  it("returns 404 for missing config file", async () => {
    const tmpDir = await freshTmpDir();
    const configDir = path.join(tmpDir, "tidal", "configs");
    await fsp.mkdir(configDir, { recursive: true });

    const result = await readConfigFile(tmpDir, "tidal", "nonexistent.yaml");
    assert.equal(result.status, 404);
  });

  it("returns 400 for path traversal attempt", async () => {
    const tmpDir = await freshTmpDir();
    const configDir = path.join(tmpDir, "tidal", "configs");
    await fsp.mkdir(configDir, { recursive: true });

    const result = await readConfigFile(tmpDir, "tidal", "../../../etc/passwd");
    assert.equal(result.status, 400);
    assert.equal(result.body.error, "Invalid filename");
  });

  it("returns 400 for non-YAML files", async () => {
    const tmpDir = await freshTmpDir();
    const configDir = path.join(tmpDir, "tidal", "configs");
    await fsp.mkdir(configDir, { recursive: true });

    const result = await readConfigFile(tmpDir, "tidal", "secrets.json");
    assert.equal(result.status, 400);
  });

  it("returns 404 for nonexistent plugin", async () => {
    const tmpDir = await freshTmpDir();

    const result = await readConfigFile(tmpDir, "nonexistent", "base_config.yaml");
    assert.equal(result.status, 404);
  });
});

describe("GET /api/plugins/:name/configs (list config files)", () => {
  it("lists all YAML files in the config directory", async () => {
    const tmpDir = await freshTmpDir();
    const configDir = path.join(tmpDir, "tidal", "configs");
    await fsp.mkdir(configDir, { recursive: true });
    await fsp.writeFile(path.join(configDir, "base_config.yaml"), "a: 1");
    await fsp.writeFile(path.join(configDir, "rl_config.yaml"), "b: 2");
    await fsp.writeFile(path.join(configDir, "notes.txt"), "ignore me");

    const result = await listConfigFiles(tmpDir, "tidal");
    assert.equal(result.status, 200);
    assert.equal(result.body.plugin, "tidal");
    assert.deepEqual(result.body.files, ["base_config.yaml", "rl_config.yaml"]);
  });

  it("returns 404 for missing plugin directory", async () => {
    const tmpDir = await freshTmpDir();

    const result = await listConfigFiles(tmpDir, "nonexistent");
    assert.equal(result.status, 404);
  });

  it("returns empty array for directory with no YAML files", async () => {
    const tmpDir = await freshTmpDir();
    const configDir = path.join(tmpDir, "tidal", "configs");
    await fsp.mkdir(configDir, { recursive: true });
    await fsp.writeFile(path.join(configDir, "readme.md"), "hello");

    const result = await listConfigFiles(tmpDir, "tidal");
    assert.equal(result.status, 200);
    assert.deepEqual(result.body.files, []);
  });
});
