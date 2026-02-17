import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import fsp from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import Fastify from "fastify";
import type { FastifyInstance } from "fastify";
import type { ServerConfig } from "../../config.js";
import gpuInstanceRoutes from "../gpu-instance.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const cleanups: string[] = [];

async function freshTmpDir(): Promise<string> {
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), "gpu-route-test-"));
  cleanups.push(dir);
  return dir;
}

after(async () => {
  for (const dir of cleanups) {
    await fsp.rm(dir, { recursive: true, force: true });
  }
});

async function buildApp(experimentsDir: string): Promise<FastifyInstance> {
  const app = Fastify({ logger: false });
  app.decorate("serverConfig", { experimentsDir } as unknown as ServerConfig);
  await app.register(gpuInstanceRoutes);
  return app;
}

// ---------------------------------------------------------------------------
// GET /api/experiments/:expId/gpu-instance
// ---------------------------------------------------------------------------

describe("GET /api/experiments/:expId/gpu-instance", () => {
  it("returns metadata when file exists", async () => {
    const tmpDir = await freshTmpDir();
    const expDir = path.join(tmpDir, "exp-gpu-001");
    await fsp.mkdir(expDir, { recursive: true });

    const meta = {
      instanceId: 31562809,
      offerId: 500,
      gpuName: "RTX A6000",
      costPerHour: 0.65,
      capturedAt: 1700000000000,
    };
    await fsp.writeFile(
      path.join(expDir, "gpu_instance.json"),
      JSON.stringify(meta),
    );

    const app = await buildApp(tmpDir);
    const resp = await app.inject({
      method: "GET",
      url: "/api/experiments/exp-gpu-001/gpu-instance",
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.expId, "exp-gpu-001");
    assert.equal(body.instance.instanceId, 31562809);
    assert.equal(body.instance.gpuName, "RTX A6000");
    assert.equal(body.instance.costPerHour, 0.65);

    await app.close();
  });

  it("returns null instance when file is missing", async () => {
    const tmpDir = await freshTmpDir();
    // No experiment directory at all

    const app = await buildApp(tmpDir);
    const resp = await app.inject({
      method: "GET",
      url: "/api/experiments/nonexistent/gpu-instance",
    });

    assert.equal(resp.statusCode, 200);
    const body = resp.json();
    assert.equal(body.expId, "nonexistent");
    assert.equal(body.instance, null);

    await app.close();
  });
});
