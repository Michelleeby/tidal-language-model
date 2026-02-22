import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { RateLimiter } from "../rate-limiter.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("RateLimiter.acquire()", () => {
  it("resolves immediately when tokens are available", async () => {
    const limiter = new RateLimiter({ capacity: 4, refillRatePerMs: 0.001 });
    const start = Date.now();
    await limiter.acquire();
    const elapsed = Date.now() - start;
    assert.ok(elapsed < 100, `Should resolve immediately, took ${elapsed}ms`);
  });

  it("waits when token bucket is exhausted", async () => {
    // 1 token, no refill → second acquire must wait
    const limiter = new RateLimiter({ capacity: 1, refillRatePerMs: 0 });
    await limiter.acquire(); // drains the only token

    let resolved = false;
    const pending = limiter.acquire().then(() => {
      resolved = true;
    });

    await sleep(20);
    assert.equal(resolved, false, "Should not resolve when no tokens remain");

    // Drain the promise to avoid unhandled rejection (refill will never happen
    // with refillRatePerMs=0 and no setTimeout scheduled — the Promise just hangs.
    // We abandon it deliberately here.)
    pending.catch(() => {});
  });

  it("tokens refill over time and unblock waiters", async () => {
    // 1 capacity, fast refill: 1 token per 5ms
    const limiter = new RateLimiter({ capacity: 1, refillRatePerMs: 1 / 5 });
    await limiter.acquire(); // drains the only token

    const start = Date.now();
    await limiter.acquire(); // should unblock after ~5ms
    const elapsed = Date.now() - start;

    // Should unblock within 200ms (loose bound for CI jitter)
    assert.ok(elapsed < 200, `Should refill and unblock within 200ms, took ${elapsed}ms`);
    // Should have waited at least a little
    assert.ok(elapsed >= 1, `Should have waited at least 1ms for refill`);
  });

  it("concurrent acquire() calls are serialized — only capacity proceed immediately", async () => {
    const capacity = 3;
    // Fast refill so the test doesn't hang waiting for stragglers
    const limiter = new RateLimiter({ capacity, refillRatePerMs: 100 });

    const resolvedOrder: number[] = [];
    const promises = Array.from({ length: capacity + 2 }, (_, i) =>
      limiter.acquire().then(() => {
        resolvedOrder.push(i);
      }),
    );

    // After a tick, exactly `capacity` callers should have resolved
    await sleep(20);
    assert.equal(
      resolvedOrder.length,
      capacity,
      `Expected exactly ${capacity} immediate resolutions, got ${resolvedOrder.length}`,
    );

    // Let remaining promises resolve (fast refill will handle them quickly)
    await Promise.all(promises);
    assert.equal(resolvedOrder.length, capacity + 2, "All callers should eventually resolve");
  });
});

describe("RateLimiter.backoff429()", () => {
  it("returns exponential backoff delays in milliseconds", () => {
    const limiter = new RateLimiter({ capacity: 4, refillRatePerMs: 0.001 });
    assert.equal(limiter.backoff429(0), 2000, "attempt 0 → 2s");
    assert.equal(limiter.backoff429(1), 4000, "attempt 1 → 4s");
    assert.equal(limiter.backoff429(2), 8000, "attempt 2 → 8s");
  });

  it("respects Retry-After header (seconds → milliseconds)", () => {
    const limiter = new RateLimiter({ capacity: 4, refillRatePerMs: 0.001 });
    assert.equal(limiter.backoff429(0, "10"), 10_000, "Retry-After: 10 → 10s");
    assert.equal(limiter.backoff429(1, "5"), 5_000, "Retry-After: 5 → 5s");
    assert.equal(limiter.backoff429(0, "0"), 0, "Retry-After: 0 → 0s");
  });

  it("ignores invalid Retry-After header and falls back to exponential", () => {
    const limiter = new RateLimiter({ capacity: 4, refillRatePerMs: 0.001 });
    assert.equal(limiter.backoff429(0, "not-a-number"), 2000);
    assert.equal(limiter.backoff429(1, ""), 4000);
  });
});
