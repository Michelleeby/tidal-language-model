import { spawn } from "node:child_process";
import fsp from "node:fs/promises";
import path from "node:path";
import type { ServerConfig } from "../config.js";
import type { GenerateRequest, GenerateResponse } from "@tidal/shared";

/**
 * Spawns Generator.py as a subprocess to generate text.
 */
export class GenerationBridge {
  constructor(private config: ServerConfig) {}

  /**
   * When the user selects an RL checkpoint as the main checkpoint, resolve
   * the best model checkpoint from the same experiment directory and use the
   * RL checkpoint as --rl-checkpoint instead.
   */
  private async resolveCheckpoints(
    req: GenerateRequest,
  ): Promise<{ modelCheckpoint: string; rlCheckpoint?: string }> {
    const basename = path.basename(req.checkpoint);
    const isRLCheckpoint = basename.startsWith("rl_checkpoint");

    let rlCheckpoint = req.rlCheckpoint;
    let modelCheckpoint = req.checkpoint;

    if (isRLCheckpoint) {
      // The user selected an RL checkpoint — find the model checkpoint
      rlCheckpoint = req.checkpoint;
      const expDir = path.dirname(req.checkpoint);
      modelCheckpoint = await this.findModelCheckpoint(expDir);
    } else if (req.gatingMode === "learned" && !rlCheckpoint) {
      // Learned mode but no RL checkpoint specified — auto-find one
      const expDir = path.dirname(req.checkpoint);
      const found = await this.findRLCheckpoint(expDir);
      if (found) rlCheckpoint = found;
    }

    return { modelCheckpoint, rlCheckpoint };
  }

  /** Find the best model checkpoint in an experiment directory. */
  private async findModelCheckpoint(expDir: string): Promise<string> {
    const entries = await fsp.readdir(expDir);
    // Prefer final model, then highest-epoch foundational checkpoint
    const finalModel = entries.find((f) => f.endsWith(".pth") && f.includes("_v") && !f.startsWith("rl_"));
    if (finalModel) return path.join(expDir, finalModel);

    const epochCheckpoints = entries
      .filter((f) => f.startsWith("checkpoint_foundational"))
      .sort()
      .reverse();
    if (epochCheckpoints.length > 0) return path.join(expDir, epochCheckpoints[0]);

    throw new Error(`No model checkpoint found in ${expDir}`);
  }

  /** Find the best RL checkpoint in an experiment directory. */
  private async findRLCheckpoint(expDir: string): Promise<string | null> {
    const entries = await fsp.readdir(expDir);
    const final = entries.find((f) => f === "rl_checkpoint_final.pth");
    if (final) return path.join(expDir, final);

    const iterCheckpoints = entries
      .filter((f) => f.startsWith("rl_checkpoint_iter_"))
      .sort()
      .reverse();
    if (iterCheckpoints.length > 0) return path.join(expDir, iterCheckpoints[0]);

    return null;
  }

  async generate(req: GenerateRequest): Promise<GenerateResponse> {
    const start = Date.now();
    const { modelCheckpoint, rlCheckpoint } = await this.resolveCheckpoints(req);

    const args = [
      "Generator.py",
      "--config",
      "configs/base_config.yaml",
      "--checkpoint",
      modelCheckpoint,
      "--prompt",
      req.prompt,
      "--max_tokens",
      String(req.maxTokens ?? 50),
      "--temperature",
      String(req.temperature ?? 0.8),
      "--top_k",
      String(req.topK ?? 50),
    ];

    if (req.gatingMode === "learned" && rlCheckpoint) {
      args.push("--rl-agent", "--rl-checkpoint", rlCheckpoint);
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
