# 0004. Lazy Disk Cache for MCP HTTP Client

**Date:** 2026-02-19
**Status:** Accepted

## Context

All Tidal training runs execute on a remote DigitalOcean droplet. The MCP
server (`@tidal/mcp`) communicates with the Fastify dashboard API over the
network for every tool call — `get_metrics`, `get_evaluation`, `get_ablation`,
etc. Each call is a full HTTP round-trip to the remote API.

When analyzing completed experiments, these calls are redundant: a completed
experiment's metrics, checkpoints, evaluation results, and RL data are
immutable. Repeated queries to the same experiment (common during analysis,
comparisons, and report generation) re-fetch identical JSON payloads every
time, including across Claude Code sessions.

The 15 MCP tool handlers all depend on a shared `TidalApiClient` interface
(`http-client.ts:23`). Adding cache logic inside each handler would scatter
the concern across 8+ files — a textbook Shotgun Surgery antipattern.

## Decision

Add a `CachingTidalApiClient` decorator that wraps any `TidalApiClient`,
implements the same interface, and transparently caches GET responses for
completed experiments to local disk. Tool handlers remain completely unaware
of the cache.

```
index.ts:  client = new CachingTidalApiClient(new FetchTidalApiClient(...), cacheDir)
```

### What gets cached

Only GET requests for per-experiment sub-resource endpoints
(`/api/experiments/:expId/*`) when the experiment is known to be completed.
Seven endpoints are cacheable: `metrics`, `status`, `rl-metrics`,
`checkpoints`, `evaluation`, `ablation`, and `gpu-instance`.

Never cached: the experiment list (`/api/experiments`), plugin endpoints,
job endpoints, report endpoints, and all POST requests.

### Completion detection

The caching client passively inspects successful API responses to learn
which experiments are done:

- `list_experiments` response: scans the `experiments` array for entries
  where `status.status === "completed"`.
- `get_status` response: checks if `status === "completed"` for the
  requested experiment.

The in-memory `completedExperiments: Set<string>` is rebuilt each session
(not persisted to disk). Cache files on disk survive across sessions but
are not read until completion is re-confirmed via the API, preventing stale
reads if an experiment ID were hypothetically reused.

### Cache storage

- **Location**: `TIDAL_CACHE_DIR` env var, defaulting to `~/.cache/tidal/`.
- **Structure**: `{cacheDir}/{expId}/{hash}.json`.
- **Key**: SHA-256 (first 16 hex chars) of `path + sorted query params`.
- **Collision guard**: the full `cacheKey` string is stored inside the JSON
  file and verified on read; a hash collision results in a cache miss, not
  corrupt data.
- **No TTL**: completed experiments are immutable by definition.
- **No new dependencies**: uses `node:fs/promises`, `node:crypto`,
  `node:path`, `node:os`.

### Files modified

- **`dashboard/packages/mcp/src/http-client.ts`**: Added
  `CachingTidalApiClient` class (~100 lines) alongside the existing
  `FetchTidalApiClient` and `TidalApiClient` interface.
- **`dashboard/packages/mcp/src/index.ts`**: 3-line change to wrap the
  `FetchTidalApiClient` in `CachingTidalApiClient`.
- **`dashboard/packages/mcp/src/__tests__/http-client.test.ts`**: 12 new
  test cases using a `spyClient` pattern and temp directories.

## Consequences

### Positive
- Repeated queries to completed experiments (the common case during analysis
  and report generation) are instant — zero network latency, no API load.
- Cache survives across Claude Code sessions, so re-analyzing the same
  experiment days later still avoids network calls.
- Zero changes to tool handler code — the decorator is invisible to
  consumers of `TidalApiClient`.
- No new runtime dependencies.

### Negative
- Disk usage grows proportionally to the number of completed experiments
  and the variety of queries made. Each cached response is a small JSON
  file (typically 1-100 KB), so this is bounded in practice.
- The `completedExperiments` set is rebuilt from scratch on each process
  start, requiring at least one `list_experiments` or `get_status` call
  before caching activates. This is acceptable since those calls happen
  naturally at the start of any MCP session.

### Neutral
- In-progress experiments are never cached, so live monitoring of active
  training runs is completely unaffected.
- If disk is full or read-only, `writeCache` fails silently and the client
  degrades to pass-through mode — identical to the pre-cache behavior.
- Corrupt cache files (e.g., from a killed process mid-write) are detected
  by JSON parse failure and treated as cache misses.

## Alternatives Considered

### Per-tool handler caching
Each of the 8+ tool handlers could maintain its own in-memory or disk cache.
Rejected because it scatters cache logic across many files (Shotgun Surgery),
requires each handler to independently track experiment completion, and makes
it easy to miss a handler or introduce inconsistencies.

### In-memory LRU cache (no disk)
A simple `Map` or LRU cache inside the client would avoid disk I/O. Rejected
because the cache would be lost on every process restart. MCP servers are
short-lived (one per Claude Code session), so cross-session persistence is
the primary value of caching. In-memory caching alone would only help within
a single session where the same query is repeated.

### HTTP-level caching (Cache-Control headers)
The Fastify API could set `Cache-Control: immutable` on responses for
completed experiments, and the client could use a caching HTTP library.
Rejected because it requires server-side changes, adds a dependency on an
HTTP caching library, and the server currently doesn't distinguish completed
vs. in-progress experiments at the response header level. The decorator
approach is entirely client-side and self-contained.

### Redis or SQLite local cache
A structured store would support TTLs, eviction, and queries. Rejected as
over-engineering — the data is small JSON files, there's no TTL requirement
(immutable data), and adding a dependency (or even SQLite via `better-sqlite3`)
increases the install footprint for marginal benefit.

## References

- Code: `dashboard/packages/mcp/src/http-client.ts` (lines 32-196)
- Code: `dashboard/packages/mcp/src/index.ts` (lines 26-30)
- Tests: `dashboard/packages/mcp/src/__tests__/http-client.test.ts`
- ADR: [0002 — Use uint16 for Tokenized Data Cache](0002-uint16-data-cache.md) (prior caching-related decision)
