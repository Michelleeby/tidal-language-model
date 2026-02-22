import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { ObjectStore } from "../object-store.js";

// ---------------------------------------------------------------------------
// Mock S3 client via constructor injection
// ---------------------------------------------------------------------------

interface SpyEntry {
  command: string;
  input: Record<string, unknown>;
}

let sendSpy: SpyEntry[] = [];
let mockSendResult: unknown = {};
let mockSendError: Error | null = null;

function makeCommand(cmdName: string) {
  const cls = class {
    readonly _name = cmdName;
    constructor(public input: Record<string, unknown>) {}
  };
  Object.defineProperty(cls, "name", { value: cmdName, writable: false });
  return cls;
}

// Exported command constructors for assertions
const Commands = {
  PutObject: makeCommand("PutObjectCommand"),
  DeleteObject: makeCommand("DeleteObjectCommand"),
  DeleteObjects: makeCommand("DeleteObjectsCommand"),
  ListObjectsV2: makeCommand("ListObjectsV2Command"),
  HeadObject: makeCommand("HeadObjectCommand"),
  GetObject: makeCommand("GetObjectCommand"),
};

function buildMockClient() {
  return {
    send: async (command: { _name: string; input: Record<string, unknown> }) => {
      sendSpy.push({ command: command._name, input: command.input });
      if (mockSendError) throw mockSendError;
      return mockSendResult;
    },
  };
}

/**
 * Build an ObjectStore with a mock S3 client.
 * We inject the mock client AND patch the module-level dynamic imports
 * by providing overrides via the clientOverride param and also injecting
 * the command constructors via the store's _commandOverrides.
 */
function makeConfiguredStore() {
  const store = new ObjectStore(
    {
      endpoint: "https://sfo3.digitaloceanspaces.com",
      region: "sfo3",
      accessKeyId: "test-key",
      secretAccessKey: "test-secret",
      bucket: "tidal-experiments",
    },
    buildMockClient() as any,
    Commands as any,
  );
  return store;
}

beforeEach(() => {
  sendSpy = [];
  mockSendResult = {};
  mockSendError = null;
});

// ---------------------------------------------------------------------------
// isConfigured()
// ---------------------------------------------------------------------------

describe("ObjectStore.isConfigured()", () => {
  it("returns false with null config", () => {
    const store = new ObjectStore(null);
    assert.equal(store.isConfigured(), false);
  });

  it("returns true with valid config", () => {
    const store = makeConfiguredStore();
    assert.equal(store.isConfigured(), true);
  });
});

// ---------------------------------------------------------------------------
// Methods throw when not configured
// ---------------------------------------------------------------------------

describe("ObjectStore — unconfigured methods", () => {
  it("putObject throws when not configured", async () => {
    const store = new ObjectStore(null);
    await assert.rejects(
      () => store.putObject("key", "body"),
      /not configured/i,
    );
  });

  it("deleteObject throws when not configured", async () => {
    const store = new ObjectStore(null);
    await assert.rejects(
      () => store.deleteObject("key"),
      /not configured/i,
    );
  });

  it("deletePrefix throws when not configured", async () => {
    const store = new ObjectStore(null);
    await assert.rejects(
      () => store.deletePrefix("prefix/"),
      /not configured/i,
    );
  });

  it("listPrefix throws when not configured", async () => {
    const store = new ObjectStore(null);
    await assert.rejects(
      () => store.listPrefix("prefix/"),
      /not configured/i,
    );
  });

  it("headObject returns exists:false when not configured", async () => {
    const store = new ObjectStore(null);
    const result = await store.headObject("key");
    assert.equal(result.exists, false);
  });
});

// ---------------------------------------------------------------------------
// putObject
// ---------------------------------------------------------------------------

describe("ObjectStore.putObject()", () => {
  it("calls PutObjectCommand with correct bucket/key", async () => {
    const store = makeConfiguredStore();
    await store.putObject("experiments/exp-1/model.pth", "data", "application/octet-stream");

    assert.equal(sendSpy.length, 1);
    assert.equal(sendSpy[0].command, "PutObjectCommand");
    assert.equal(sendSpy[0].input.Bucket, "tidal-experiments");
    assert.equal(sendSpy[0].input.Key, "experiments/exp-1/model.pth");
    assert.equal(sendSpy[0].input.ContentType, "application/octet-stream");
  });
});

// ---------------------------------------------------------------------------
// deletePrefix
// ---------------------------------------------------------------------------

describe("ObjectStore.deletePrefix()", () => {
  it("lists objects then batch deletes", async () => {
    mockSendResult = {
      Contents: [
        { Key: "experiments/exp-1/model.pth" },
        { Key: "experiments/exp-1/metadata.json" },
      ],
      IsTruncated: false,
    };

    const store = makeConfiguredStore();
    await store.deletePrefix("experiments/exp-1/");

    // First call: ListObjectsV2Command
    assert.equal(sendSpy[0].command, "ListObjectsV2Command");
    assert.equal(sendSpy[0].input.Bucket, "tidal-experiments");
    assert.equal(sendSpy[0].input.Prefix, "experiments/exp-1/");

    // Second call: DeleteObjectsCommand
    assert.equal(sendSpy[1].command, "DeleteObjectsCommand");
  });

  it("is a no-op when prefix has no objects", async () => {
    mockSendResult = { Contents: [], IsTruncated: false };
    const store = makeConfiguredStore();
    await store.deletePrefix("experiments/empty/");
    // Only one call (list), no delete
    assert.equal(sendSpy.length, 1);
    assert.equal(sendSpy[0].command, "ListObjectsV2Command");
  });
});

// ---------------------------------------------------------------------------
// headObject
// ---------------------------------------------------------------------------

describe("ObjectStore.headObject()", () => {
  it("returns exists:true when object found", async () => {
    mockSendResult = { ContentLength: 1024 };
    const store = makeConfiguredStore();
    const result = await store.headObject("experiments/exp-1/model.pth");
    assert.equal(result.exists, true);
    assert.equal(result.sizeBytes, 1024);
  });

  it("returns exists:false on NoSuchKey error", async () => {
    const err = new Error("NoSuchKey") as Error & { name: string };
    err.name = "NoSuchKey";
    mockSendError = err;
    const store = makeConfiguredStore();
    const result = await store.headObject("nonexistent.pth");
    assert.equal(result.exists, false);
  });

  it("returns exists:false on 404 Not Found", async () => {
    const err = new Error("Not Found") as Error & { $metadata?: { httpStatusCode: number } };
    (err as any).$metadata = { httpStatusCode: 404 };
    mockSendError = err;
    const store = makeConfiguredStore();
    const result = await store.headObject("nonexistent.pth");
    assert.equal(result.exists, false);
  });
});
