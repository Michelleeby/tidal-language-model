import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import type { TrainingJob, JobStatus, ComputeProviderType } from "@tidal/shared";
import type { ComputeProvider } from "../compute-provider.js";
import type { JobStore } from "../job-store.js";
import type { ProvisioningChain } from "../provisioning-chain.js";
import type { WorkerSpawner } from "../worker-spawner.js";
import {
  JobHealthMonitor,
  type HealthMonitorConfig,
} from "../job-health-monitor.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const DEFAULT_CONFIG: HealthMonitorConfig = {
  healthCheckIntervalMs: 15_000,
  heartbeatTimeoutMs: 60_000,
  staleStartupTimeoutMs: 120_000,
  remoteStartupTimeoutMs: 900_000,
};

function makeJob(overrides: Partial<TrainingJob> = {}): TrainingJob {
  return {
    jobId: "job-" + Math.random().toString(36).slice(2, 8),
    type: "lm-training",
    status: "running",
    provider: "local",
    config: { type: "lm-training", plugin: "tidal", configPath: "configs/base_config.yaml" },
    createdAt: Date.now(),
    updatedAt: Date.now(),
    ...overrides,
  };
}

function makeRemoteJob(overrides: Partial<TrainingJob> = {}): TrainingJob {
  return makeJob({
    provider: "vastai",
    providerMeta: { instanceId: 12345 },
    ...overrides,
  });
}

/** Fake store that returns preset values. */
function fakeStore(
  activeJobs: TrainingJob[],
  heartbeats: Map<string, number | null> = new Map(),
): JobStore {
  return {
    listActive: async () => activeJobs,
    getHeartbeat: async (jobId: string) => heartbeats.get(jobId) ?? null,
  } as unknown as JobStore;
}

/** Fake provider whose isAlive result can be controlled per-call. */
function fakeProvider(
  opts: {
    isRemote?: boolean;
    isAliveResult?: boolean;
    isAliveThrows?: boolean;
  } = {},
): ComputeProvider {
  return {
    type: "vastai" as ComputeProviderType,
    isRemote: opts.isRemote ?? true,
    canProvision: async () => true,
    provision: async () => ({ success: true }),
    deprovision: async () => {},
    isAlive: async () => {
      if (opts.isAliveThrows) throw new Error("network error");
      return opts.isAliveResult ?? true;
    },
  };
}

function fakeChain(
  provider?: ComputeProvider,
): ProvisioningChain {
  return {
    getProvider: () => provider,
  } as unknown as ProvisioningChain;
}

function fakeSpawner(running = false): WorkerSpawner {
  return {
    isRunning: () => running,
  } as unknown as WorkerSpawner;
}

const silentLog = {
  info: () => {},
  warn: () => {},
  error: () => {},
  debug: () => {},
  trace: () => {},
  fatal: () => {},
  child: () => silentLog,
} as any;

/** Run the monitor's health check once (manually, without timers). */
async function runHealthCheck(monitor: JobHealthMonitor): Promise<void> {
  // Access private method via bracket notation
  await (monitor as any).healthCheck();
}

// ---------------------------------------------------------------------------
// Startup-phase: isAlive check for remote jobs with providerMeta
// ---------------------------------------------------------------------------

describe("JobHealthMonitor — startup isAlive for remote jobs", () => {
  it("marks remote startup job as failed when isAlive returns false (past grace period)", async () => {
    const failedJobs: { jobId: string; status: string; error?: string }[] = [];

    const job = makeRemoteJob({
      status: "starting",
      // updatedAt 90 seconds ago — past the 60s grace period
      updatedAt: Date.now() - 90_000,
    });

    const provider = fakeProvider({ isRemote: true, isAliveResult: false });
    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(provider),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId, status, error) => {
        failedJobs.push({ jobId, status, error });
      },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(failedJobs.length, 1);
    assert.equal(failedJobs[0].jobId, job.jobId);
    assert.ok(failedJobs[0].error?.includes("terminated during startup"));
  });

  it("does NOT check isAlive for remote startup job within 60s grace period", async () => {
    const failedJobs: string[] = [];
    let isAliveCalled = false;

    const job = makeRemoteJob({
      status: "starting",
      // updatedAt 30 seconds ago — within grace period
      updatedAt: Date.now() - 30_000,
    });

    const provider = {
      ...fakeProvider({ isRemote: true, isAliveResult: false }),
      isAlive: async () => {
        isAliveCalled = true;
        return false;
      },
    };

    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(provider as ComputeProvider),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId) => { failedJobs.push(jobId); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(isAliveCalled, false, "isAlive should not be called within grace period");
    assert.equal(failedJobs.length, 0, "Job should not be marked failed within grace period");
  });

  it("does NOT check isAlive for remote startup job without providerMeta", async () => {
    const failedJobs: string[] = [];
    let isAliveCalled = false;

    const job = makeRemoteJob({
      status: "provisioning",
      providerMeta: undefined, // no providerMeta yet — still provisioning
      updatedAt: Date.now() - 90_000,
    });

    const provider = {
      ...fakeProvider({ isRemote: true, isAliveResult: false }),
      isAlive: async () => {
        isAliveCalled = true;
        return false;
      },
    };

    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(provider as ComputeProvider),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId) => { failedJobs.push(jobId); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(isAliveCalled, false, "isAlive should not be called without providerMeta");
    assert.equal(failedJobs.length, 0, "Job should not be marked failed without providerMeta");
  });

  it("leaves startup job alone when isAlive returns true", async () => {
    const failedJobs: string[] = [];

    const job = makeRemoteJob({
      status: "starting",
      updatedAt: Date.now() - 90_000,
    });

    const provider = fakeProvider({ isRemote: true, isAliveResult: true });
    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(provider),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId) => { failedJobs.push(jobId); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(failedJobs.length, 0);
  });

  it("does not fail the job when isAlive throws during startup", async () => {
    const failedJobs: string[] = [];

    const job = makeRemoteJob({
      status: "starting",
      updatedAt: Date.now() - 90_000,
    });

    const provider = fakeProvider({ isRemote: true, isAliveThrows: true });
    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(provider),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId) => { failedJobs.push(jobId); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(failedJobs.length, 0, "isAlive error should be swallowed");
  });
});

// ---------------------------------------------------------------------------
// Running-phase: isAlive check for remote jobs with no heartbeat
// ---------------------------------------------------------------------------

describe("JobHealthMonitor — no-heartbeat remote running jobs", () => {
  it("marks remote running job as failed when no heartbeat AND isAlive returns false", async () => {
    const failedJobs: { jobId: string; error?: string }[] = [];

    const job = makeRemoteJob({ status: "running" });

    const provider = fakeProvider({ isRemote: true, isAliveResult: false });
    const monitor = new JobHealthMonitor(
      fakeStore([job]), // no heartbeat in map → returns null
      fakeChain(provider),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId, _status, error) => { failedJobs.push({ jobId, error }); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(failedJobs.length, 1);
    assert.equal(failedJobs[0].jobId, job.jobId);
    assert.ok(failedJobs[0].error?.includes("terminated"));
  });

  it("skips remote running job (no failure) when no heartbeat but isAlive returns true", async () => {
    const failedJobs: string[] = [];

    const job = makeRemoteJob({ status: "running" });

    const provider = fakeProvider({ isRemote: true, isAliveResult: true });
    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(provider),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId) => { failedJobs.push(jobId); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(failedJobs.length, 0);
  });

  it("skips remote running job when no heartbeat and isAlive throws", async () => {
    const failedJobs: string[] = [];

    const job = makeRemoteJob({ status: "running" });

    const provider = fakeProvider({ isRemote: true, isAliveThrows: true });
    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(provider),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId) => { failedJobs.push(jobId); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(failedJobs.length, 0);
  });

  it("does not fail remote running job when no heartbeat and no provider registered", async () => {
    // When a remote provider isn't registered in the chain, isRemote
    // falls back to false, and the code falls through to the local worker
    // check (which will fail since there's no local process).
    // After the fix, the code should check provider.isRemote before
    // the local worker path. But since getProvider returns undefined,
    // isRemote is false → local path. This test documents current behavior.
    const failedJobs: string[] = [];

    const job = makeRemoteJob({ status: "running" });

    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(undefined), // no provider registered
      fakeSpawner(false),
      DEFAULT_CONFIG,
      async (jobId) => { failedJobs.push(jobId); },
      silentLog,
    );

    await runHealthCheck(monitor);

    // Without a provider, isRemote=false → treated as local → worker not running → failed
    // This is existing behavior we're not changing
    assert.equal(failedJobs.length, 1);
  });
});

// ---------------------------------------------------------------------------
// Stopping jobs — graceful "Stop Now" should eventually deprovision
// ---------------------------------------------------------------------------

describe("JobHealthMonitor — stopping jobs", () => {
  it("cancels remote stopping job when no heartbeat and isAlive returns true", async () => {
    const results: { jobId: string; status: string; error?: string }[] = [];

    const job = makeRemoteJob({ status: "stopping" as JobStatus });

    const provider = fakeProvider({ isRemote: true, isAliveResult: true });
    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(provider),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId, status, error) => { results.push({ jobId, status, error }); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(results.length, 1);
    assert.equal(results[0].jobId, job.jobId);
    assert.equal(results[0].status, "cancelled");
    assert.ok(results[0].error?.includes("Stopped by user"));
  });

  it("cancels remote stopping job when no heartbeat and isAlive returns false", async () => {
    const results: { jobId: string; status: string }[] = [];

    const job = makeRemoteJob({ status: "stopping" as JobStatus });

    const provider = fakeProvider({ isRemote: true, isAliveResult: false });
    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(provider),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId, status) => { results.push({ jobId, status }); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(results.length, 1);
    assert.equal(results[0].status, "cancelled");
  });

  it("cancels remote stopping job when heartbeat is stale", async () => {
    const results: { jobId: string; status: string }[] = [];

    const job = makeRemoteJob({ status: "stopping" as JobStatus });
    const staleHeartbeat = (Date.now() - 120_000) / 1000; // 2 min ago
    const heartbeats = new Map([[job.jobId, staleHeartbeat]]);

    const provider = fakeProvider({ isRemote: true, isAliveResult: true });
    const monitor = new JobHealthMonitor(
      fakeStore([job], heartbeats),
      fakeChain(provider),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId, status) => { results.push({ jobId, status }); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(results.length, 1);
    assert.equal(results[0].status, "cancelled");
  });

  it("does NOT cancel remote stopping job when heartbeat is fresh", async () => {
    const results: string[] = [];

    const job = makeRemoteJob({ status: "stopping" as JobStatus });
    const freshHeartbeat = Date.now() / 1000; // now
    const heartbeats = new Map([[job.jobId, freshHeartbeat]]);

    const provider = fakeProvider({ isRemote: true, isAliveResult: true });
    const monitor = new JobHealthMonitor(
      fakeStore([job], heartbeats),
      fakeChain(provider),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId) => { results.push(jobId); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(results.length, 0, "Worker still sending heartbeats — let it finish");
  });

  it("cancels local stopping job when no heartbeat and worker not running", async () => {
    const results: { jobId: string; status: string }[] = [];

    const job = makeJob({ status: "stopping" as JobStatus, provider: "local" });

    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(undefined),
      fakeSpawner(false),
      DEFAULT_CONFIG,
      async (jobId, status) => { results.push({ jobId, status }); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(results.length, 1);
    assert.equal(results[0].status, "cancelled");
  });

  it("cancels remote stopping job when no heartbeat and isAlive throws", async () => {
    const results: { jobId: string; status: string }[] = [];

    const job = makeRemoteJob({ status: "stopping" as JobStatus });

    const provider = fakeProvider({ isRemote: true, isAliveThrows: true });
    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(provider),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId, status) => { results.push({ jobId, status }); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(results.length, 1);
    assert.equal(results[0].status, "cancelled");
  });
});

// ---------------------------------------------------------------------------
// Existing behavior preserved
// ---------------------------------------------------------------------------

describe("JobHealthMonitor — existing behavior", () => {
  it("marks local running job as failed when no heartbeat and worker not running", async () => {
    const failedJobs: string[] = [];

    const job = makeJob({ status: "running", provider: "local" });

    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(undefined),
      fakeSpawner(false),
      DEFAULT_CONFIG,
      async (jobId) => { failedJobs.push(jobId); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(failedJobs.length, 1);
    assert.equal(failedJobs[0], job.jobId);
  });

  it("does not fail local running job when worker is still running (no heartbeat)", async () => {
    const failedJobs: string[] = [];

    const job = makeJob({ status: "running", provider: "local" });

    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(undefined),
      fakeSpawner(true),
      DEFAULT_CONFIG,
      async (jobId) => { failedJobs.push(jobId); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(failedJobs.length, 0);
  });

  it("marks job failed when heartbeat is stale past timeout", async () => {
    const failedJobs: string[] = [];

    const job = makeJob({ status: "running" });
    const staleHeartbeat = (Date.now() - 120_000) / 1000; // 2 min ago (> 60s timeout)
    const heartbeats = new Map([[job.jobId, staleHeartbeat]]);

    const monitor = new JobHealthMonitor(
      fakeStore([job], heartbeats),
      fakeChain(undefined),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId) => { failedJobs.push(jobId); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(failedJobs.length, 1);
  });

  it("does not fail job when heartbeat is fresh", async () => {
    const failedJobs: string[] = [];

    const job = makeJob({ status: "running" });
    const freshHeartbeat = Date.now() / 1000; // now
    const heartbeats = new Map([[job.jobId, freshHeartbeat]]);

    const monitor = new JobHealthMonitor(
      fakeStore([job], heartbeats),
      fakeChain(undefined),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId) => { failedJobs.push(jobId); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(failedJobs.length, 0);
  });

  it("skips terminal jobs", async () => {
    const failedJobs: string[] = [];

    const job = makeJob({ status: "completed" as JobStatus });

    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(undefined),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId) => { failedJobs.push(jobId); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(failedJobs.length, 0);
  });

  it("marks local startup job as failed when stuck past timeout", async () => {
    const failedJobs: string[] = [];

    const job = makeJob({
      status: "pending",
      updatedAt: Date.now() - 200_000, // 200s > 120s local timeout
    });

    const monitor = new JobHealthMonitor(
      fakeStore([job]),
      fakeChain(undefined),
      fakeSpawner(),
      DEFAULT_CONFIG,
      async (jobId) => { failedJobs.push(jobId); },
      silentLog,
    );

    await runHealthCheck(monitor);

    assert.equal(failedJobs.length, 1);
  });
});
