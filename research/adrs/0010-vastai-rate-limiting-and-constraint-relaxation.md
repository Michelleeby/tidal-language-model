# 0010. Vast.ai Rate Limiting and Progressive Constraint Relaxation

**Date:** 2026-02-23
**Status:** Accepted

## Context

During concurrent ADR-0008 experiment launches, two production failures exposed missing infrastructure in the Vast.ai provisioning path:

1. **429 rate limiting** — Vast.ai's API enforces a ~4.5 req/sec threshold. When multiple jobs provision simultaneously, the burst of search and create-instance calls exceeds this limit, causing 429 responses and permanent provisioning failures (no retry logic existed).

2. **Brittle offer constraints** — The original provider hardcoded a single set of network and reliability thresholds (`800 Mbps` down/up, `0.99` reliability). On Feb 12, 2026, two back-to-back production fixes were needed within minutes:
   - `8ca49d3` reduced reliability from `0.99` → `0.95` after "No suitable vast.ai GPU offers found"
   - `e5cb861` reduced bandwidth from `800` → `400` Mbps after the same failure recurred

   These ad-hoc fixes restored provisioning but left a single set of loose constraints with no attempt to prefer higher-quality instances when available.

The dashboard already had a **Redis-backed token bucket** rate limiter (`dashboard/packages/server/src/plugins/rate-limit.ts`, commit `24ddb93`) for protecting the `/api/generate` endpoint from user abuse. However, this serves a fundamentally different purpose — it rate-limits inbound requests from distributed clients using IP-based keys in Redis. The Vast.ai rate limiting problem is about throttling outbound requests from a single server process.

## Decision

### In-process token bucket for Vast.ai API calls

Use an **in-process token bucket** (`dashboard/packages/server/src/services/providers/rate-limiter.ts`) to throttle all outbound Vast.ai API calls. Configuration: 4-token capacity, 1 token/sec refill rate (conservative vs. the 4.5 req/sec threshold).

**Why not Redis-backed**: All Vast.ai API calls originate from a single Fastify server process. There is no distributed coordination problem — the `VastAIProvider` instance making the calls is the only caller. An in-process bucket eliminates a Redis round-trip on every API call and avoids a Redis availability dependency for provisioning (Redis is optional for the dashboard and degrades gracefully elsewhere).

The two rate limiters in the system serve distinct roles:

| Limiter | Purpose | Scope | Backend |
|---|---|---|---|
| `plugins/rate-limit.ts` | Protect `/api/generate` from user abuse | Inbound, distributed clients, IP-keyed | Redis (Lua script) |
| `providers/rate-limiter.ts` | Throttle outbound Vast.ai API calls | Outbound, single process | In-process memory |

If the dashboard is later scaled to multiple server processes that share Vast.ai provisioning, the in-process limiter would need to be replaced with a distributed one (Redis-backed or similar). At current scale (single server), in-process is the correct choice.

### 429 retry with exponential backoff

The `VastAIProvider.fetch()` wrapper retries up to 3 times on 429 responses, using exponential backoff (2s, 4s, 8s) or the `Retry-After` header when present. All Vast.ai API calls (`findOffers`, `createInstance`, `deprovision`, `isAlive`) go through this wrapper.

### 3-tier progressive constraint relaxation

Replace the single hardcoded constraint set with a 3-tier relaxation schedule for offer search:

| Tier | Bandwidth (down/up) | Reliability | Origin |
|---|---|---|---|
| 0 (strict) | 800 Mbps | 0.99 | Original values — high-quality instances preferred |
| 1 (relaxed) | 400 Mbps | 0.97 | Midpoint between strict and production-validated floor |
| 2 (loose) | 200 Mbps | 0.95 | Floor values validated in production (`8ca49d3`, `e5cb861`) |

**How the values were derived**: Tier 0 is the original specification that works when Vast.ai inventory is plentiful. Tier 2 is the floor that was empirically validated on Feb 12 when strict constraints yielded zero offers. Tier 1 is the arithmetic midpoint, providing an intermediate fallback before reaching the floor. The bandwidth steps halve at each tier (800 → 400 → 200); reliability steps down by 0.02 (0.99 → 0.97 → 0.95).

Provisioning attempts each tier in order, falling through to the next when no offers match or all offers in a tier fail. This prefers higher-quality instances while guaranteeing the same success rate as the post-fix configuration.

The tiers are configurable via `VastAIProviderConfig.relaxationTiers` for testing and future tuning.

## Consequences

### Positive
- Eliminates 429-induced provisioning failures under concurrent job launches
- Prefers high-bandwidth, high-reliability instances when available, without failing when they aren't
- Zero Redis dependency for Vast.ai API throttling — provisioning works even if Redis is down
- Constraint tiers are configurable and testable (injected via constructor)

### Negative
- In-process rate limiter state is lost on server restart (tokens reset to full capacity — acceptable since this is a burst allowance, not a persistent quota)
- Progressive relaxation increases provisioning latency in low-inventory scenarios (up to 3 search rounds before finding an offer)

### Neutral
- The 3-tier schedule codifies values that were previously ad-hoc production fixes — future maintainers can trace the tier values to commits `8ca49d3` and `e5cb861`
- If the dashboard scales to multiple processes, the in-process limiter must be replaced with a distributed implementation

## Alternatives Considered

### Redis-backed rate limiting for Vast.ai calls
Reuse the existing Redis Lua-script token bucket from `plugins/rate-limit.ts`. This would provide distributed coordination but adds a Redis round-trip on every outbound API call and makes provisioning depend on Redis availability. Since all Vast.ai calls originate from a single process, the coordination benefit is zero and the costs are real.

### Single loose constraint set (status quo post-fix)
Keep the Feb 12 fix values (400 Mbps, 0.95 reliability) as the only constraint set. Simpler, but always provisions the cheapest available instance even when high-quality options exist. The tiered approach gets better instances when inventory allows.

### Dynamic constraint adjustment based on Vast.ai inventory
Query the Vast.ai API for current inventory levels and adjust constraints dynamically. More adaptive, but adds complexity and extra API calls (which themselves are rate-limited). The 3-tier static schedule is simple, predictable, and covers the observed failure modes.

## References

- PR: [#28 — Robust Vast.ai provisioning, rate limiting & log fix](https://github.com/Michelleeby/tidal-language-model/pull/28)
- Code: `dashboard/packages/server/src/services/providers/rate-limiter.ts`
- Code: `dashboard/packages/server/src/services/providers/vastai-provider.ts`
- Code: `dashboard/packages/server/src/plugins/rate-limit.ts` (existing Redis-backed limiter)
- Commit `8ca49d3`: Reduced reliability from 0.99 → 0.95 after production failure
- Commit `e5cb861`: Reduced bandwidth from 800 → 400 Mbps after production failure
- Commit `24ddb93`: Original Redis-backed rate limiting for dashboard endpoints
