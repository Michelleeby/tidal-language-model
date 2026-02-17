import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { TidalApiClient, ApiResult } from "../http-client.js";
import {
  handleListPlugins,
  handleGetPluginManifest,
  handleGetConfig,
} from "../tools/plugin-tools.js";

// ---------------------------------------------------------------------------
// Mock client factory
// ---------------------------------------------------------------------------

function okClient<T>(data: T): TidalApiClient {
  return {
    get: async () => ({ ok: true, data, status: 200 }) as ApiResult<never>,
    post: async () => ({ ok: true, data: {}, status: 200 }) as ApiResult<never>,
  };
}

function errClient(status: number, error: string): TidalApiClient {
  return {
    get: async () => ({ ok: false, error, status }) as ApiResult<never>,
    post: async () => ({ ok: false, error, status }) as ApiResult<never>,
  };
}

// ---------------------------------------------------------------------------
// handleListPlugins
// ---------------------------------------------------------------------------

describe("handleListPlugins", () => {
  it("returns plugin list", async () => {
    const data = {
      plugins: [
        {
          name: "tidal",
          displayName: "Tidal LM",
          version: "1.0.0",
          trainingPhases: [{ id: "lm", displayName: "LM Training" }],
          generationModes: [{ id: "none", displayName: "No gating" }],
        },
      ],
    };
    const result = await handleListPlugins(okClient(data));

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.plugins.length, 1);
    assert.equal(parsed.plugins[0].name, "tidal");
  });

  it("returns error on failure", async () => {
    const result = await handleListPlugins(errClient(500, "Server error"));
    assert.equal(result.isError, true);
    assert.match(result.content[0].text as string, /Server error/);
  });
});

// ---------------------------------------------------------------------------
// handleGetPluginManifest
// ---------------------------------------------------------------------------

describe("handleGetPluginManifest", () => {
  it("returns full plugin manifest", async () => {
    const data = {
      plugin: {
        name: "tidal",
        displayName: "Tidal LM",
        version: "1.0.0",
        description: "A tidal language model",
        trainingPhases: [],
        checkpointPatterns: [],
        generation: {},
        metrics: {},
        redis: {},
        infrastructure: {},
      },
    };
    const result = await handleGetPluginManifest(okClient(data), {
      pluginName: "tidal",
    });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.plugin.name, "tidal");
  });

  it("verifies correct path is used", async () => {
    let capturedPath = "";
    const client: TidalApiClient = {
      get: async (path) => {
        capturedPath = path;
        return { ok: true, data: { plugin: null }, status: 200 } as ApiResult<never>;
      },
      post: async () => ({ ok: true, data: {}, status: 200 }) as ApiResult<never>,
    };

    await handleGetPluginManifest(client, { pluginName: "tidal" });
    assert.equal(capturedPath, "/api/plugins/tidal");
  });

  it("defaults to 'tidal' when pluginName is omitted", async () => {
    let capturedPath = "";
    const client: TidalApiClient = {
      get: async (path) => {
        capturedPath = path;
        return { ok: true, data: { plugin: null }, status: 200 } as ApiResult<never>;
      },
      post: async () => ({ ok: true, data: {}, status: 200 }) as ApiResult<never>,
    };

    await handleGetPluginManifest(client, {});
    assert.equal(capturedPath, "/api/plugins/tidal");
  });
});

// ---------------------------------------------------------------------------
// handleGetConfig
// ---------------------------------------------------------------------------

describe("handleGetConfig", () => {
  it("returns config file content", async () => {
    const data = {
      filename: "base_config.yaml",
      content: "VOCAB_SIZE: 50257\nEMBED_DIM: 512",
    };
    const result = await handleGetConfig(okClient(data), {
      pluginName: "tidal",
      filename: "base_config.yaml",
    });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.filename, "base_config.yaml");
    assert.match(parsed.content, /VOCAB_SIZE/);
  });

  it("verifies correct path is used", async () => {
    let capturedPath = "";
    const client: TidalApiClient = {
      get: async (path) => {
        capturedPath = path;
        return {
          ok: true,
          data: { filename: "x.yaml", content: "" },
          status: 200,
        } as ApiResult<never>;
      },
      post: async () => ({ ok: true, data: {}, status: 200 }) as ApiResult<never>,
    };

    await handleGetConfig(client, {
      pluginName: "tidal",
      filename: "rl_config.yaml",
    });
    assert.equal(capturedPath, "/api/plugins/tidal/configs/rl_config.yaml");
  });

  it("returns error on missing config", async () => {
    const result = await handleGetConfig(errClient(404, "Config not found"), {
      pluginName: "tidal",
      filename: "missing.yaml",
    });
    assert.equal(result.isError, true);
  });
});
