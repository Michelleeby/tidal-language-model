import fsp from "node:fs/promises";

export interface ObjectStoreConfig {
  endpoint: string;        // DO_SPACES_ENDPOINT
  region: string;          // DO_SPACES_REGION
  accessKeyId: string;     // DO_SPACES_KEY
  secretAccessKey: string; // DO_SPACES_SECRET
  bucket: string;          // DO_SPACES_BUCKET
}

export interface HeadResult {
  exists: boolean;
  sizeBytes?: number;
}

// Minimal interfaces for testability
interface S3ClientLike {
  send(command: unknown): Promise<unknown>;
}

interface S3Commands {
  PutObject: new (input: Record<string, unknown>) => object;
  DeleteObject: new (input: Record<string, unknown>) => object;
  DeleteObjects: new (input: Record<string, unknown>) => object;
  ListObjectsV2: new (input: Record<string, unknown>) => object;
  HeadObject: new (input: Record<string, unknown>) => object;
  GetObject: new (input: Record<string, unknown>) => object;
}

/**
 * S3-compatible object storage abstraction for DigitalOcean Spaces.
 *
 * Follows the same graceful degradation pattern as Redis:
 * - isConfigured() returns false when Spaces env vars are missing
 * - All methods throw descriptive errors when not configured
 *   (except headObject which returns { exists: false })
 *
 * Accept optional client and command overrides for unit testing without
 * requiring the AWS SDK to be installed.
 */
export class ObjectStore {
  private config: ObjectStoreConfig | null;
  private _client: S3ClientLike | null = null;
  private _commandOverrides: S3Commands | null;

  constructor(
    config: ObjectStoreConfig | null,
    clientOverride?: S3ClientLike,
    commandOverrides?: S3Commands,
  ) {
    this.config = config;
    this._client = clientOverride ?? null;
    this._commandOverrides = commandOverrides ?? null;
  }

  isConfigured(): boolean {
    return this.config !== null;
  }

  private requireConfig(): ObjectStoreConfig {
    if (!this.config) {
      throw new Error("ObjectStore not configured — set DO_SPACES_* environment variables");
    }
    return this.config;
  }

  private async getClient(): Promise<S3ClientLike> {
    if (this._client) return this._client;

    const cfg = this.requireConfig();
    const { S3Client } = await import("@aws-sdk/client-s3");
    this._client = new S3Client({
      endpoint: cfg.endpoint,
      region: cfg.region,
      credentials: {
        accessKeyId: cfg.accessKeyId,
        secretAccessKey: cfg.secretAccessKey,
      },
      forcePathStyle: false,
    });
    return this._client;
  }

  private async getCommands(): Promise<S3Commands> {
    if (this._commandOverrides) return this._commandOverrides;
    const sdk = await import("@aws-sdk/client-s3");
    return {
      PutObject: sdk.PutObjectCommand,
      DeleteObject: sdk.DeleteObjectCommand,
      DeleteObjects: sdk.DeleteObjectsCommand,
      ListObjectsV2: sdk.ListObjectsV2Command,
      HeadObject: sdk.HeadObjectCommand,
      GetObject: sdk.GetObjectCommand,
    };
  }

  /**
   * Upload a small object (string or Buffer) to Spaces.
   */
  async putObject(key: string, body: string | Buffer | Uint8Array, contentType?: string): Promise<void> {
    const cfg = this.requireConfig();
    const client = await this.getClient();
    const cmds = await this.getCommands();

    await client.send(
      new cmds.PutObject({
        Bucket: cfg.bucket,
        Key: key,
        Body: body,
        ...(contentType ? { ContentType: contentType } : {}),
      }),
    );
  }

  /**
   * Upload a large file using multipart upload.
   */
  async putLargeFile(key: string, filePath: string): Promise<void> {
    const cfg = this.requireConfig();
    const client = await this.getClient();
    const { Upload } = await import("@aws-sdk/lib-storage");
    const { createReadStream } = await import("node:fs");

    const upload = new Upload({
      client,
      params: {
        Bucket: cfg.bucket,
        Key: key,
        Body: createReadStream(filePath),
      },
    });
    await upload.done();
  }

  /**
   * Download an object as a Buffer.
   */
  async getObject(key: string): Promise<Buffer> {
    const cfg = this.requireConfig();
    const client = await this.getClient();
    const cmds = await this.getCommands();

    const resp = await client.send(new cmds.GetObject({ Bucket: cfg.bucket, Key: key })) as {
      Body?: { transformToByteArray(): Promise<Uint8Array> };
    };

    if (!resp.Body) throw new Error(`Empty body for key: ${key}`);
    const bytes = await resp.Body.transformToByteArray();
    return Buffer.from(bytes);
  }

  /**
   * Delete a single object.
   */
  async deleteObject(key: string): Promise<void> {
    const cfg = this.requireConfig();
    const client = await this.getClient();
    const cmds = await this.getCommands();

    await client.send(new cmds.DeleteObject({ Bucket: cfg.bucket, Key: key }));
  }

  /**
   * Delete all objects under a prefix (list + batch delete).
   */
  async deletePrefix(prefix: string): Promise<void> {
    const cfg = this.requireConfig();
    const client = await this.getClient();
    const cmds = await this.getCommands();

    let continuationToken: string | undefined;
    do {
      const listResp = await client.send(
        new cmds.ListObjectsV2({
          Bucket: cfg.bucket,
          Prefix: prefix,
          ...(continuationToken ? { ContinuationToken: continuationToken } : {}),
        }),
      ) as { Contents?: Array<{ Key?: string }>; IsTruncated?: boolean; NextContinuationToken?: string };

      const objects = (listResp.Contents ?? []).filter((o) => o.Key);
      if (objects.length > 0) {
        await client.send(
          new cmds.DeleteObjects({
            Bucket: cfg.bucket,
            Delete: {
              Objects: objects.map((o) => ({ Key: o.Key! })),
              Quiet: true,
            },
          }),
        );
      }

      continuationToken = listResp.IsTruncated ? listResp.NextContinuationToken : undefined;
    } while (continuationToken);
  }

  /**
   * List all keys under a prefix.
   */
  async listPrefix(prefix: string): Promise<string[]> {
    const cfg = this.requireConfig();
    const client = await this.getClient();
    const cmds = await this.getCommands();

    const keys: string[] = [];
    let continuationToken: string | undefined;

    do {
      const listResp = await client.send(
        new cmds.ListObjectsV2({
          Bucket: cfg.bucket,
          Prefix: prefix,
          ...(continuationToken ? { ContinuationToken: continuationToken } : {}),
        }),
      ) as { Contents?: Array<{ Key?: string }>; IsTruncated?: boolean; NextContinuationToken?: string };

      for (const obj of listResp.Contents ?? []) {
        if (obj.Key) keys.push(obj.Key);
      }

      continuationToken = listResp.IsTruncated ? listResp.NextContinuationToken : undefined;
    } while (continuationToken);

    return keys;
  }

  /**
   * Check if a key exists. Returns { exists: false } when not configured.
   */
  async headObject(key: string): Promise<HeadResult> {
    if (!this.config) return { exists: false };

    try {
      const client = await this.getClient();
      const cmds = await this.getCommands();

      const resp = await client.send(
        new cmds.HeadObject({ Bucket: this.config.bucket, Key: key }),
      ) as { ContentLength?: number };

      return { exists: true, sizeBytes: resp.ContentLength };
    } catch (err: unknown) {
      const e = err as { name?: string; $metadata?: { httpStatusCode?: number } };
      if (e.name === "NoSuchKey" || e.$metadata?.httpStatusCode === 404) {
        return { exists: false };
      }
      throw err;
    }
  }

  /**
   * Download an object directly to disk.
   */
  async downloadToFile(key: string, destPath: string): Promise<void> {
    const buf = await this.getObject(key);
    await fsp.writeFile(destPath, buf);
  }
}
