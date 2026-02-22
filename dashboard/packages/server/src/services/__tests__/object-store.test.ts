import { describe, it, mock, before } from "node:test";
import assert from "node:assert/strict";

// ---------------------------------------------------------------------------
// Mock S3 SDK before any imports that use it
// ---------------------------------------------------------------------------

interface MockCommand {
  input: Record<string, unknown>;
}

// We'll track calls to the mock S3Client
let sendSpy: Array<{ command: string; input: Record<string, unknown> }> = [];
let mockSendResult: unknown = {};
let mockSendError: Error | null = null;

class MockS3Client {
  async send(command: MockCommand) {
    const commandName = command.constructor.name;
    sendSpy.push({ command: commandName, input: command.input });
    if (mockSendError) throw mockSendError;
    return mockSendResult;
  }
}

function makeCommand(name: string) {
  return class {
    constructor(public input: Record<string, unknown>) {}
    get [Symbol.toStringTag]() { return name; }
  };
}

const MockPutObjectCommand = makeCommand("PutObjectCommand");
const MockDeleteObjectCommand = makeCommand("DeleteObjectCommand");
const MockDeleteObjectsCommand = makeCommand("DeleteObjectsCommand");
const MockListObjectsV2Command = makeCommand("ListObjectsV2Command");
const MockHeadObjectCommand = makeCommand("HeadObjectCommand");
const MockGetObjectCommand = makeCommand("GetObjectCommand");

before(() => {
  mock.module("@aws-sdk/client-s3", {
    namedExports: {
      S3Client: MockS3Client,
      PutObjectCommand: MockPutObjectCommand,
      DeleteObjectCommand: MockDeleteObjectCommand,
      DeleteObjectsCommand: MockDeleteObjectsCommand,
      ListObjectsV2Command: MockListObjectsV2Command,
      HeadObjectCommand: MockHeadObjectCommand,
      GetObjectCommand: MockGetObjectCommand,
    },
  });
  mock.module("@aws-sdk/lib-storage", {
    namedExports: {
      Upload: class MockUpload {
        constructor(public opts: Record<string, unknown>) {}
        async done() { return {}; }
      },
    },
  });
});

// Import AFTER mock is registered
const { ObjectStore } = await import("../object-store.js");

function resetMock() {
  sendSpy = [];
  mockSendResult = {};
  mockSendError = null;
}

function makeConfiguredStore() {
  return new ObjectStore({
    endpoint: "https://sfo3.digitaloceanspaces.com",
    region: "sfo3",
    accessKeyId: "test-key",
    secretAccessKey: "test-secret",
    bucket: "tidal-experiments",
  });
}

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
    resetMock();
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
    resetMock();
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
    resetMock();
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
    resetMock();
    mockSendResult = { ContentLength: 1024 };
    const store = makeConfiguredStore();
    const result = await store.headObject("experiments/exp-1/model.pth");
    assert.equal(result.exists, true);
    assert.equal(result.sizeBytes, 1024);
  });

  it("returns exists:false on NoSuchKey error", async () => {
    resetMock();
    const err = new Error("NoSuchKey") as Error & { name: string };
    err.name = "NoSuchKey";
    mockSendError = err;
    const store = makeConfiguredStore();
    const result = await store.headObject("nonexistent.pth");
    assert.equal(result.exists, false);
  });

  it("returns exists:false on 404 Not Found", async () => {
    resetMock();
    const err = new Error("Not Found") as Error & { $metadata?: { httpStatusCode: number } };
    (err as any).$metadata = { httpStatusCode: 404 };
    mockSendError = err;
    const store = makeConfiguredStore();
    const result = await store.headObject("nonexistent.pth");
    assert.equal(result.exists, false);
  });
});
