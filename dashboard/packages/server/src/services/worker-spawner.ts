import { spawn, type ChildProcess } from "node:child_process";
import type { FastifyBaseLogger } from "fastify";
import type { JobConfig } from "@tidal/shared";

export class WorkerSpawner {
  private processes = new Map<string, ChildProcess>();

  constructor(
    private projectRoot: string,
    private pythonBin: string,
    private redisUrl: string,
    private log: FastifyBaseLogger,
  ) {}

  spawnLocal(jobId: string, _config: JobConfig): void {
    const child = spawn(
      this.pythonBin,
      ["worker_agent.py", "--job-id", jobId, "--redis-url", this.redisUrl],
      {
        cwd: this.projectRoot,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
          REDIS_URL: this.redisUrl,
          TIDAL_JOB_ID: jobId,
        },
        stdio: ["ignore", "pipe", "pipe"],
        // Start a new process group so we can kill worker + all children
        detached: true,
      },
    );

    this.processes.set(jobId, child);

    child.stdout?.on("data", (data: Buffer) => {
      this.log.info({ jobId }, `[worker] ${data.toString().trimEnd()}`);
    });

    child.stderr?.on("data", (data: Buffer) => {
      this.log.warn({ jobId }, `[worker:err] ${data.toString().trimEnd()}`);
    });

    child.on("exit", (code, signal) => {
      this.log.info({ jobId, code, signal }, "Worker process exited");
      this.processes.delete(jobId);
    });

    child.on("error", (err) => {
      this.log.error({ jobId, err }, "Worker process error");
      this.processes.delete(jobId);
    });
  }

  /** Kill the entire process group (worker + training subprocess). */
  kill(jobId: string): void {
    const child = this.processes.get(jobId);
    if (!child || child.exitCode !== null) return;

    const pid = child.pid;
    if (!pid) {
      this.log.warn({ jobId }, "No PID for worker, cannot kill");
      return;
    }

    this.log.info({ jobId, pid }, "Sending SIGTERM to worker process group");
    try {
      // Negative PID kills the entire process group
      process.kill(-pid, "SIGTERM");
    } catch {
      // Process may already be dead
    }

    // Grace period: SIGKILL the group after 10s
    const timer = setTimeout(() => {
      if (child.exitCode === null) {
        this.log.warn({ jobId }, "Worker group did not exit, sending SIGKILL");
        try {
          process.kill(-pid, "SIGKILL");
        } catch {
          // Already dead
        }
      }
    }, 10_000);

    child.on("exit", () => clearTimeout(timer));
  }

  isRunning(jobId: string): boolean {
    const child = this.processes.get(jobId);
    return child != null && child.exitCode === null;
  }

  cleanup(): void {
    for (const [jobId] of this.processes) {
      this.kill(jobId);
    }
  }
}
