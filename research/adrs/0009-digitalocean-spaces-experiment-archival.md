# 0009. DigitalOcean Spaces for Experiment Archival

**Date:** 2026-02-23
**Status:** Accepted

## Context

The dashboard host is at ~60% disk capacity. Experiment directories accumulate `.pth` checkpoint files (often 100–250 MB each) that are rarely accessed after training completes but must be retained for reproducibility and checkpoint restoration. Continuing to store all experiment artifacts on the host's local disk is unsustainable — a few more training runs will push storage past safe operating margins.

The dashboard already follows a graceful-degradation pattern for optional services (Redis degrades to disk-only metrics). A similar pattern is needed for long-term storage: checkpoint archival should be offloaded to a service designed for durable, inexpensive object storage, while metadata and real-time metrics stay local for fast reads.

## Decision

Use **DigitalOcean Spaces** (S3-compatible object storage) as the archival tier for experiment checkpoints and report version history. The integration is opt-in via five `DO_SPACES_*` environment variables and degrades gracefully when unconfigured.

### Components

**`ObjectStore`** (`dashboard/packages/server/src/services/object-store.ts`): S3-compatible abstraction over `@aws-sdk/client-s3`. Accepts `null` config for unconfigured mode — `isConfigured()` returns `false`, write methods throw descriptively, `headObject` returns `{ exists: false }`. Lazy-loads the AWS SDK to avoid import cost when Spaces is disabled.

**`SpacesArchiver`** (`dashboard/packages/server/src/services/spaces-archiver.ts`): Two-phase archival of experiment `.pth` files:
1. **Upload phase**: uploads all `.pth` files to `experiments/{expId}/` in Spaces, writes `_archive_manifest.json` with `state: "uploading"`
2. **Verify phase**: HEAD-verifies each uploaded file, deletes local `.pth` copies only after verification, marks manifest `state: "complete"`

Small metadata files (`metadata.json`, `config.yaml`, `gpu_instance.json`) and real-time metrics directories (`dashboard_metrics/`, `rl_metrics/`) are never archived — they stay local for fast dashboard reads.

**`ArchiveScheduler`** (`dashboard/packages/server/src/services/archive-scheduler.ts`): Background poller (default 5-minute interval) that discovers completed experiments without archive manifests and triggers `SpacesArchiver.archiveExperiment()`. Only archives experiments with `status: "completed"` and no active job heartbeat.

**`ReportRepository`** (`dashboard/packages/server/src/services/report-repository.ts`): Write-through cache for dashboard reports. `autoSave()` writes SQLite only (debounced real-time persistence); `save()` writes SQLite AND Spaces with versioned snapshots (max 5 versions). `load()` falls back from SQLite to Spaces. When Spaces is unconfigured, `save()` behaves like `autoSave()`.

**`ExperimentDeleter`** (`dashboard/packages/server/src/services/experiment-deleter.ts`): Cascading deletion across disk, Redis, SQLite, and Spaces. Reads the archive manifest before disk deletion to clean up Spaces objects for archived experiments.

**`objectStorePlugin`** (`dashboard/packages/server/src/plugins/object-store.ts`): Fastify plugin that decorates the server with an `ObjectStore` instance — configured or unconfigured based on environment variables.

### Key prefix layout in Spaces

```
experiments/{expId}/checkpoint_*.pth
experiments/{expId}/rl_agent_*.pth
reports/{reportId}/current.json
reports/{reportId}/v{timestamp}.json
```

## Consequences

### Positive
- Frees local disk from checkpoint accumulation — only metadata and metrics stay on host
- Archived experiments remain accessible via `SpacesArchiver.retrieveFile()` for checkpoint restoration
- Report versioning provides undo history beyond the client-side `HistoryManager` session scope
- Fully opt-in — zero behavioral change when `DO_SPACES_*` vars are unset
- Follows the established graceful-degradation pattern (same as Redis)

### Negative
- Adds `@aws-sdk/client-s3` and `@aws-sdk/lib-storage` as server dependencies (~2 MB)
- Archived checkpoint retrieval requires a network round-trip (cold restore is slower than local disk)
- DigitalOcean Spaces incurs storage costs ($5/250 GB + $0.01/GB transfer)

### Neutral
- Partial Spaces configuration (e.g., setting 3 of 5 env vars) is treated as a startup error to prevent silent misconfiguration
- The archive manifest file (`_archive_manifest.json`) becomes part of the experiment directory contract

## Alternatives Considered

### Local disk rotation (delete old experiments)
Deleting old experiments frees disk but destroys reproducibility. There is no way to recover a checkpoint once deleted. This trades storage pressure for permanent data loss.

### NFS / network-attached block storage
Block storage (e.g., DigitalOcean Volumes) would extend local capacity but still ties storage to the host's lifecycle and region. Object storage is cheaper per GB, has built-in redundancy, and decouples storage from compute.

### Self-hosted MinIO
MinIO provides S3-compatible storage but requires managing another service (deployment, backups, monitoring). DigitalOcean Spaces is a managed service with the same S3 API, eliminating operational overhead.

## References

- PR: [#29 — DigitalOcean Spaces integration and experiment lifecycle management](https://github.com/Michelleeby/tidal-language-model/pull/29)
- Code: `dashboard/packages/server/src/services/object-store.ts`
- Code: `dashboard/packages/server/src/services/spaces-archiver.ts`
- Code: `dashboard/packages/server/src/services/archive-scheduler.ts`
- Code: `dashboard/packages/server/src/services/report-repository.ts`
- Code: `dashboard/packages/server/src/services/experiment-deleter.ts`
- Related ADR: [0004 — Lazy disk cache for MCP HTTP client](../adrs/0004-lazy-disk-cache-for-mcp-http-client.md)
