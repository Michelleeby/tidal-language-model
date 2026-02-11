import type { FastifyReply } from "fastify";
import type Redis from "ioredis";
import type { SSEEvent, TrainingJob } from "@tidal/shared";

interface Client {
  reply: FastifyReply;
  expId: string;
  lastStep: number;
}

interface GlobalClient {
  reply: FastifyReply;
}

/**
 * Manages SSE connections, polling Redis for updates and pushing events.
 */
export class SSEManager {
  private clients = new Map<string, Set<Client>>();
  private globalClients = new Set<GlobalClient>();
  private timer: ReturnType<typeof setInterval> | null = null;
  private subscriber: Redis | null = null;

  constructor(
    private redis: Redis | null,
    private pollIntervalMs: number = 2000,
  ) {
    this.subscribeToJobUpdates();
  }

  start() {
    if (this.timer) return;
    this.timer = setInterval(() => this.poll(), this.pollIntervalMs);
  }

  stop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.subscriber) {
      this.subscriber.disconnect();
      this.subscriber = null;
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
      if (this.clients.size === 0 && this.globalClients.size === 0) {
        this.stop();
      }
    });
  }

  addGlobalClient(reply: FastifyReply) {
    const client: GlobalClient = { reply };
    this.globalClients.add(client);

    // Send initial heartbeat
    this.sendRawEvent(reply, {
      type: "heartbeat",
      data: { timestamp: Date.now() },
    });

    this.start();

    reply.raw.on("close", () => {
      this.globalClients.delete(client);
      if (this.clients.size === 0 && this.globalClients.size === 0) {
        this.stop();
      }
    });
  }

  broadcastJobUpdate(job: TrainingJob) {
    const event: SSEEvent = { type: "job-update", data: job };
    for (const client of this.globalClients) {
      this.sendRawEvent(client.reply, event);
    }
  }

  private async subscribeToJobUpdates() {
    if (!this.redis) return;
    try {
      // Create a dedicated subscriber connection (ioredis requirement)
      this.subscriber = this.redis.duplicate();
      await this.subscriber.subscribe("tidal:job:updates");
      this.subscriber.on("message", async (_channel, message) => {
        try {
          const { jobId } = JSON.parse(message);
          if (!jobId || !this.redis) return;
          const raw = await this.redis.hget("tidal:jobs", jobId);
          if (raw) {
            this.broadcastJobUpdate(JSON.parse(raw));
          }
        } catch {
          // Non-fatal
        }
      });
    } catch {
      // Redis pub/sub not available — global clients rely on polling
    }
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

    // Heartbeat for global clients
    for (const client of this.globalClients) {
      this.sendRawEvent(client.reply, {
        type: "heartbeat",
        data: { timestamp: Date.now() },
      });
    }
  }

  private sendEvent(client: Client, event: SSEEvent) {
    this.sendRawEvent(client.reply, event);
  }

  private sendRawEvent(reply: FastifyReply, event: SSEEvent) {
    try {
      reply.raw.write(
        `event: ${event.type}\ndata: ${JSON.stringify(event.data)}\n\n`,
      );
    } catch {
      // Client may have disconnected
    }
  }
}
