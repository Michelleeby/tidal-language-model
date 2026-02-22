/**
 * Token bucket rate limiter for Vast.ai API calls.
 *
 * In-process implementation — no Redis needed. A single server process
 * makes all Vast.ai API calls, so a shared in-memory bucket is sufficient.
 *
 * Concurrency safety: Node.js is single-threaded, but concurrent async
 * callers can interleave between the token check and decrement. A pending-
 * waiters queue ensures only one caller proceeds per token: when acquire()
 * finds no tokens it enqueues a resolver; when tokens refill the drainWaiters
 * loop resolves them one at a time.
 */

export interface RateLimiterConfig {
  /** Maximum token capacity (burst allowance). */
  capacity: number;
  /** Tokens added per millisecond. E.g. 1/1000 = 1 token/sec. */
  refillRatePerMs: number;
}

export class RateLimiter {
  private tokens: number;
  private lastRefillAt: number;
  private readonly capacity: number;
  private readonly refillRatePerMs: number;
  private readonly waiters: Array<() => void> = [];

  constructor(config: RateLimiterConfig) {
    this.capacity = config.capacity;
    this.refillRatePerMs = config.refillRatePerMs;
    this.tokens = config.capacity;
    this.lastRefillAt = Date.now();
  }

  /** Refill tokens based on elapsed time since last refill. */
  private refill(): void {
    const now = Date.now();
    const elapsed = now - this.lastRefillAt;
    this.tokens = Math.min(this.capacity, this.tokens + elapsed * this.refillRatePerMs);
    this.lastRefillAt = now;
  }

  /**
   * Drain the waiters queue: for each available token, resolve one waiter.
   * Called synchronously, so no interleaving between check and decrement.
   */
  private drainWaiters(): void {
    while (this.tokens >= 1 && this.waiters.length > 0) {
      this.tokens -= 1;
      const resolve = this.waiters.shift()!;
      resolve();
    }
  }

  /**
   * Acquire one token, waiting asynchronously if none are available.
   * Safe for concurrent callers — uses a waiters queue to prevent races.
   */
  async acquire(): Promise<void> {
    this.refill();
    if (this.tokens >= 1) {
      this.tokens -= 1;
      return;
    }

    // No tokens available — enqueue and wait for a refill callback
    return new Promise<void>((resolve) => {
      this.waiters.push(resolve);
      if (this.refillRatePerMs > 0) {
        // Schedule a refill check at the earliest time a new token arrives
        const tokensNeeded = 1 - this.tokens;
        const msUntilToken = Math.ceil(tokensNeeded / this.refillRatePerMs);
        setTimeout(() => {
          this.refill();
          this.drainWaiters();
        }, msUntilToken);
      }
      // If refillRatePerMs === 0, the caller waits forever (caller's problem).
    });
  }

  /**
   * Return the number of milliseconds to wait before retrying a 429 response.
   *
   * @param attempt  Zero-based retry count (0 = first retry).
   * @param retryAfterHeader  Optional value from the Retry-After response header (seconds).
   */
  backoff429(attempt: number, retryAfterHeader?: string): number {
    if (retryAfterHeader !== undefined && retryAfterHeader !== "") {
      const seconds = parseInt(retryAfterHeader, 10);
      if (!isNaN(seconds)) {
        return seconds * 1000;
      }
    }
    // Exponential: 2s, 4s, 8s, ...
    return Math.pow(2, attempt + 1) * 1000;
  }
}
