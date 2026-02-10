import { spawn } from "node:child_process";
import type { ServerConfig } from "../config.js";
import type { GenerateRequest, GenerateResponse } from "@tidal/shared";

/**
 * Spawns Generator.py as a subprocess to generate text.
 */
export class GenerationBridge {
  constructor(private config: ServerConfig) {}

  async generate(req: GenerateRequest): Promise<GenerateResponse> {
    const start = Date.now();
    const args = [
      "Generator.py",
      "--config",
      "configs/base_config.yaml",
      "--checkpoint",
      req.checkpoint,
      "--prompt",
      req.prompt,
      "--max_tokens",
      String(req.maxTokens ?? 50),
      "--temperature",
      String(req.temperature ?? 0.8),
      "--top_k",
      String(req.topK ?? 50),
    ];

    if (req.gatingMode === "learned" && req.rlCheckpoint) {
      args.push("--rl-agent", "--rl-checkpoint", req.rlCheckpoint);
    }

    const text = await this.runPython(args);
    return {
      text: text.trim(),
      tokensGenerated: text.trim().split(/\s+/).length,
      elapsedMs: Date.now() - start,
    };
  }

  private runPython(args: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
      const proc = spawn(this.config.pythonBin, args, {
        cwd: this.config.projectRoot,
        timeout: 30_000,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      });

      let stdout = "";
      let stderr = "";

      proc.stdout.on("data", (chunk) => {
        stdout += chunk;
      });
      proc.stderr.on("data", (chunk) => {
        stderr += chunk;
      });

      proc.on("close", (code) => {
        if (code === 0) {
          resolve(stdout);
        } else {
          reject(new Error(`Generator.py exited with code ${code}: ${stderr}`));
        }
      });

      proc.on("error", reject);
    });
  }
}
