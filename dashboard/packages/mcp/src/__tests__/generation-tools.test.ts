import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { TidalApiClient, ApiResult } from "../http-client.js";
import { handleGenerateText } from "../tools/generation-tools.js";

// ---------------------------------------------------------------------------
// Mock client factory
// ---------------------------------------------------------------------------

function okClient<T>(data: T): TidalApiClient {
  return {
    get: async () => ({ ok: true, data: {}, status: 200 }) as ApiResult<never>,
    post: async () => ({ ok: true, data, status: 200 }) as ApiResult<never>,
  };
}

function errClient(status: number, error: string): TidalApiClient {
  return {
    get: async () => ({ ok: false, error, status }) as ApiResult<never>,
    post: async () => ({ ok: false, error, status }) as ApiResult<never>,
  };
}

// ---------------------------------------------------------------------------
// handleGenerateText
// ---------------------------------------------------------------------------

describe("handleGenerateText", () => {
  it("sends POST and returns generated text", async () => {
    const data = {
      text: "Once upon a time there was a little cat",
      tokensGenerated: 9,
      elapsedMs: 150,
    };
    const result = await handleGenerateText(okClient(data), {
      checkpoint: "model.pth",
      prompt: "Once upon a time",
    });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.equal(parsed.text, "Once upon a time there was a little cat");
    assert.equal(parsed.tokensGenerated, 9);
    assert.equal(result.isError, undefined);
  });

  it("returns text with trajectory when gating is enabled", async () => {
    const data = {
      text: "Once upon a time in a land far away",
      tokensGenerated: 9,
      elapsedMs: 200,
      trajectory: {
        gateSignals: [[0.5, 0.7, 0.3]],
        effects: [{ temperature: 0.8, repetition_penalty: 1.1, top_k: 40, top_p: 0.9 }],
        tokenIds: [7454],
        tokenTexts: ["Once"],
      },
    };
    const result = await handleGenerateText(okClient(data), {
      checkpoint: "model.pth",
      prompt: "Once upon a time",
      gatingMode: "learned",
      rlCheckpoint: "rl_agent.pth",
    });

    const parsed = JSON.parse(result.content[0].text as string);
    assert.ok(parsed.trajectory);
    assert.equal(parsed.trajectory.gateSignals.length, 1);
  });

  it("sends all optional parameters in POST body", async () => {
    let capturedBody: unknown;
    const client: TidalApiClient = {
      get: async () => ({ ok: true, data: {}, status: 200 }) as ApiResult<never>,
      post: async (_path, body) => {
        capturedBody = body;
        return {
          ok: true,
          data: { text: "", tokensGenerated: 0, elapsedMs: 0 },
          status: 200,
        } as ApiResult<never>;
      },
    };

    await handleGenerateText(client, {
      checkpoint: "model.pth",
      prompt: "Hello",
      maxTokens: 100,
      temperature: 0.7,
      topK: 40,
      gatingMode: "learned",
      rlCheckpoint: "rl.pth",
    });

    const body = capturedBody as Record<string, unknown>;
    assert.equal(body.checkpoint, "model.pth");
    assert.equal(body.prompt, "Hello");
    assert.equal(body.maxTokens, 100);
    assert.equal(body.temperature, 0.7);
    assert.equal(body.topK, 40);
    assert.equal(body.gatingMode, "learned");
    assert.equal(body.rlCheckpoint, "rl.pth");
  });

  it("returns error on failure", async () => {
    const result = await handleGenerateText(
      errClient(500, "Generation failed"),
      { checkpoint: "model.pth", prompt: "Hello" },
    );
    assert.equal(result.isError, true);
    assert.match(result.content[0].text as string, /Generation failed/);
  });
});
