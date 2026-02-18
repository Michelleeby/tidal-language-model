import { spawn } from "node:child_process";
import { accessSync } from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import type { ServerConfig } from "../config.js";
import type { GenerateRequest, GenerateResponse, AnalyzeRequest, AnalyzeResponse, PluginManifest } from "@tidal/shared";

/**
 * Convert a manifest glob pattern to a RegExp for matching filenames.
 */
function globToRegex(glob: string): RegExp {
  const escaped = glob.replace(/[.+^${}()|[\]\\]/g, "\\$&");
  const pattern = escaped.replace(/\*/g, ".*");
  return new RegExp(`^${pattern}$`);
}

/**
 * Dual-mode generation bridge:
 * - If INFERENCE_URL is set -> HTTP POST to the inference sidecar (production/Docker)
 * - Else if pythonBin exists -> subprocess spawn (local dev)
 * - Else -> 503 error
 *
 * Uses the plugin manifest to determine CLI args, checkpoint patterns,
 * and gating modes instead of hardcoded values.
 */
export class GenerationBridge {
  private inferenceUrl: string | null;
  private subprocessAvailable: boolean;
  private manifest: PluginManifest | undefined;

  constructor(
    private config: ServerConfig,
    manifest?: PluginManifest,
  ) {
    this.inferenceUrl = config.inferenceUrl;
    this.subprocessAvailable = false;
    this.manifest = manifest;

    if (!this.inferenceUrl) {
      try {
        accessSync(config.pythonBin);
        this.subprocessAvailable = true;
      } catch {
        // No Python binary available
      }
    }
  }

  get available(): boolean {
    return !!this.inferenceUrl || this.subprocessAvailable;
  }

  /**
   * When the user selects an RL checkpoint as the main checkpoint, resolve
   * the best model checkpoint from the same experiment directory and use the
   * RL checkpoint as --rl-checkpoint instead.
   */
  private async resolveCheckpoints(
    req: GenerateRequest,
  ): Promise<{ modelCheckpoint: string; rlCheckpoint?: string }> {
    const basename = path.basename(req.checkpoint);
    const isRLCheckpoint = this.isRLCheckpointFile(basename);

    let rlCheckpoint = req.rlCheckpoint;
    let modelCheckpoint = req.checkpoint;

    if (isRLCheckpoint) {
      // The user selected an RL checkpoint -- find the model checkpoint
      rlCheckpoint = req.checkpoint;
      const expDir = path.dirname(req.checkpoint);
      modelCheckpoint = await this.findModelCheckpoint(expDir);
    } else if (req.gatingMode === "learned" && !rlCheckpoint) {
      // Learned mode but no RL checkpoint specified -- auto-find one
      const expDir = path.dirname(req.checkpoint);
      const found = await this.findRLCheckpoint(expDir);
      if (found) rlCheckpoint = found;
    }

    return { modelCheckpoint, rlCheckpoint };
  }

  /** Check if a filename matches any RL checkpoint pattern from the manifest. */
  private isRLCheckpointFile(filename: string): boolean {
    const patterns = this.manifest?.generation.rlCheckpointPatterns ?? [];
    return patterns.some((glob) => globToRegex(glob).test(filename));
  }

  /** Check if a filename matches any model checkpoint pattern from the manifest. */
  private isModelCheckpointFile(filename: string): boolean {
    const patterns = this.manifest?.generation.modelCheckpointPatterns ?? [];
    return patterns.some((glob) => globToRegex(glob).test(filename));
  }

  /** Find the best model checkpoint in an experiment directory. */
  private async findModelCheckpoint(expDir: string): Promise<string> {
    const entries = await fsp.readdir(expDir);
    // Sort descending to get highest-numbered checkpoint first
    const modelFiles = entries
      .filter((f) => f.endsWith(".pth") && this.isModelCheckpointFile(f))
      .sort()
      .reverse();

    if (modelFiles.length > 0) return path.join(expDir, modelFiles[0]);

    throw new Error(`No model checkpoint found in ${expDir}`);
  }

  /** Find the best RL checkpoint in an experiment directory. */
  private async findRLCheckpoint(expDir: string): Promise<string | null> {
    const entries = await fsp.readdir(expDir);
    const rlFiles = entries
      .filter((f) => f.endsWith(".pth") && this.isRLCheckpointFile(f))
      .sort()
      .reverse();

    return rlFiles.length > 0 ? path.join(expDir, rlFiles[0]) : null;
  }

  async generate(req: GenerateRequest): Promise<GenerateResponse> {
    if (!this.available) {
      throw new Error(
        "Text generation is not available on this host — no Python environment or inference service found. " +
        "Generation requires either INFERENCE_URL or a local Python environment with model checkpoints.",
      );
    }

    const start = Date.now();
    const { modelCheckpoint, rlCheckpoint } = await this.resolveCheckpoints(req);

    if (this.inferenceUrl) {
      return this.generateViaHttp(req, modelCheckpoint, rlCheckpoint, start);
    }
    return this.generateViaSubprocess(req, modelCheckpoint, rlCheckpoint, start);
  }

  private async generateViaHttp(
    req: GenerateRequest,
    modelCheckpoint: string,
    rlCheckpoint: string | undefined,
    start: number,
  ): Promise<GenerateResponse> {
    const res = await fetch(`${this.inferenceUrl}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        checkpoint: modelCheckpoint,
        prompt: req.prompt,
        maxTokens: req.maxTokens ?? 50,
        temperature: req.temperature ?? 0.8,
        topK: req.topK ?? 50,
        gatingMode: req.gatingMode ?? "none",
        rlCheckpoint: rlCheckpoint,
        creativity: req.creativity,
        focus: req.focus,
        stability: req.stability,
      }),
      signal: AbortSignal.timeout(120_000),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(
        (body as { error?: string }).error ?? `Inference sidecar returned ${res.status}`,
      );
    }

    return (await res.json()) as GenerateResponse;
  }

  /**
   * Resolve the config path for generation: prefer the config.yaml saved in the
   * experiment directory (copied there by Main.py), fall back to manifest default.
   */
  private async resolveConfigPath(checkpointPath: string): Promise<string> {
    const expDir = path.dirname(checkpointPath);
    const expConfig = path.join(expDir, "config.yaml");
    try {
      await fsp.access(expConfig);
      return expConfig;
    } catch {
      return this.manifest?.generation.defaultConfigPath ?? "configs/base_config.yaml";
    }
  }

  private async generateViaSubprocess(
    req: GenerateRequest,
    modelCheckpoint: string,
    rlCheckpoint: string | undefined,
    start: number,
  ): Promise<GenerateResponse> {
    const gen = this.manifest?.generation;
    const argMap = gen?.args ?? {};
    const configPath = await this.resolveConfigPath(modelCheckpoint);
    const entrypoint = gen?.entrypoint ?? "Generator.py";

    const args = [entrypoint];

    // Build CLI args from manifest arg mapping
    if (argMap.config) args.push(argMap.config, configPath);
    if (argMap.checkpoint) args.push(argMap.checkpoint, modelCheckpoint);
    if (argMap.prompt) args.push(argMap.prompt, req.prompt);
    if (argMap.maxTokens) args.push(argMap.maxTokens, String(req.maxTokens ?? 50));
    if (argMap.temperature) args.push(argMap.temperature, String(req.temperature ?? 0.8));
    if (argMap.topK) args.push(argMap.topK, String(req.topK ?? 50));

    if (req.gatingMode === "learned" && rlCheckpoint) {
      if (argMap.rlAgent) args.push(argMap.rlAgent);
      if (argMap.rlCheckpoint) args.push(argMap.rlCheckpoint, rlCheckpoint);
    }

    const text = await this.runPython(args);
    return {
      text: text.trim(),
      tokensGenerated: text.trim().split(/\s+/).length,
      elapsedMs: Date.now() - start,
    };
  }

  async analyzeTrajectories(req: AnalyzeRequest): Promise<AnalyzeResponse> {
    if (!this.inferenceUrl) {
      throw new Error(
        "Trajectory analysis requires the inference sidecar (INFERENCE_URL). " +
        "Subprocess mode is not supported for batch analysis.",
      );
    }

    const res = await fetch(`${this.inferenceUrl}/analyze-trajectories`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal: AbortSignal.timeout(300_000),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(
        (body as { error?: string }).error ?? `Inference sidecar returned ${res.status}`,
      );
    }

    return (await res.json()) as AnalyzeResponse;
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
          const entrypoint = this.manifest?.generation.entrypoint ?? "Generator.py";
          reject(new Error(`${entrypoint} exited with code ${code}: ${stderr}`));
        }
      });

      proc.on("error", reject);
    });
  }
}
