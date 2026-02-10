import type { FastifyReply } from "fastify";
import type Redis from "ioredis";
import type { SSEEvent } from "@tidal/shared";

interface Client {
  reply: FastifyReply;
  expId: string;
  lastStep: number;
}

/**
 * Manages SSE connections, polling Redis for updates and pushing events.
 */
export class SSEManager {
  private clients = new Map<string, Set<Client>>();
  private timer: ReturnType<typeof setInterval> | null = null;

  constructor(
    private redis: Redis | null,
    private pollIntervalMs: number = 2000,
  ) {}

  start() {
    if (this.timer) return;
    this.timer = setInterval(() => this.poll(), this.pollIntervalMs);
  }

  stop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  addClient(expId: string, reply: FastifyReply) {
    const client: Client = { reply, expId, lastStep: -1 };

    if (!this.clients.has(expId)) {
      this.clients.set(expId, new Set());
    }
    this.clients.get(expId)!.add(client);

    // Send initial heartbeat
    this.sendEvent(client, {
      type: "heartbeat",
      data: { timestamp: Date.now() },
    });

    // Start polling if not already
    this.start();

    // Clean up on close
    reply.raw.on("close", () => {
      this.clients.get(expId)?.delete(client);
      if (this.clients.get(expId)?.size === 0) {
        this.clients.delete(expId);
      }
      if (this.clients.size === 0) {
        this.stop();
      }
    });
  }

  private async poll() {
    if (!this.redis) return;

    for (const [expId, clients] of this.clients) {
      try {
        // Check latest metrics
        const metricsRaw = await this.redis.get(
          `tidal:metrics:${expId}:latest`,
        );
        if (metricsRaw) {
          const metrics = JSON.parse(metricsRaw);
          for (const client of clients) {
            if (metrics.step > client.lastStep) {
              this.sendEvent(client, { type: "metrics", data: metrics });
              client.lastStep = metrics.step;
            }
          }
        }

        // Check status
        const statusRaw = await this.redis.get(`tidal:status:${expId}`);
        if (statusRaw) {
          const status = JSON.parse(statusRaw);
          for (const client of clients) {
            this.sendEvent(client, { type: "status", data: status });
          }
        }

        // Check RL metrics
        const rlRaw = await this.redis.get(`tidal:rl:${expId}:latest`);
        if (rlRaw) {
          for (const client of clients) {
            this.sendEvent(client, {
              type: "rl-metrics",
              data: JSON.parse(rlRaw),
            });
          }
        }
      } catch {
        // Non-fatal polling error
      }

      // Send heartbeat
      for (const client of clients) {
        this.sendEvent(client, {
          type: "heartbeat",
          data: { timestamp: Date.now() },
        });
      }
    }
  }

  private sendEvent(client: Client, event: SSEEvent) {
    try {
      client.reply.raw.write(
        `event: ${event.type}\ndata: ${JSON.stringify(event.data)}\n\n`,
      );
    } catch {
      // Client may have disconnected
    }
  }
}
