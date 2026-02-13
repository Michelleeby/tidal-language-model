import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { CheckpointPattern } from "@tidal/shared";
import { classifyCheckpoint } from "../checkpoint-classifier.js";

// ---------------------------------------------------------------------------
// Patterns matching the tidal manifest
// ---------------------------------------------------------------------------

const PATTERNS: CheckpointPattern[] = [
  {
    phase: "foundational",
    glob: "checkpoint_foundational_epoch_*.pth",
    epochCapture: "epoch_(\\d+)",
  },
  {
    phase: "rl",
    glob: "rl_checkpoint_iter_*.pth",
    epochCapture: "iter_(\\d+)",
  },
  {
    phase: "rl",
    glob: "rl_checkpoint_final.pth",
  },
  {
    phase: "final",
    glob: "*_v*.pth",
    excludePrefix: "rl_",
  },
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("classifyCheckpoint", () => {
  it("classifies foundational checkpoint and extracts epoch", () => {
    const result = classifyCheckpoint(
      "checkpoint_foundational_epoch_3.pth",
      PATTERNS,
    );
    assert.equal(result.phase, "foundational");
    assert.equal(result.epoch, 3);
  });

  it("classifies RL iteration checkpoint and extracts iteration", () => {
    const result = classifyCheckpoint(
      "rl_checkpoint_iter_500.pth",
      PATTERNS,
    );
    assert.equal(result.phase, "rl");
    assert.equal(result.epoch, 500);
  });

  it("classifies RL final checkpoint (no epoch)", () => {
    const result = classifyCheckpoint(
      "rl_checkpoint_final.pth",
      PATTERNS,
    );
    assert.equal(result.phase, "rl");
    assert.equal(result.epoch, undefined);
  });

  it("classifies final model checkpoint (version pattern)", () => {
    const result = classifyCheckpoint(
      "transformer-lm_v1.0.0.pth",
      PATTERNS,
    );
    assert.equal(result.phase, "final");
    assert.equal(result.epoch, undefined);
  });

  it("excludePrefix prevents rl_ files from matching *_v* pattern", () => {
    // An RL checkpoint with _v in the name should match rl pattern first,
    // but if it somehow only matched the *_v* pattern, excludePrefix blocks it
    const result = classifyCheckpoint(
      "rl_checkpoint_v2.pth",
      PATTERNS,
    );
    // Does not match rl_checkpoint_iter_* or rl_checkpoint_final.pth,
    // and *_v*.pth has excludePrefix: "rl_", so falls to unknown
    assert.equal(result.phase, "unknown");
  });

  it("returns unknown for unrecognized filenames", () => {
    const result = classifyCheckpoint("random_file.pth", PATTERNS);
    assert.equal(result.phase, "unknown");
    assert.equal(result.epoch, undefined);
  });

  it("returns unknown for non-.pth files", () => {
    const result = classifyCheckpoint("config.yaml", PATTERNS);
    assert.equal(result.phase, "unknown");
  });

  it("handles empty patterns array", () => {
    const result = classifyCheckpoint(
      "checkpoint_foundational_epoch_1.pth",
      [],
    );
    assert.equal(result.phase, "unknown");
  });

  it("extracts large epoch numbers", () => {
    const result = classifyCheckpoint(
      "checkpoint_foundational_epoch_12345.pth",
      PATTERNS,
    );
    assert.equal(result.phase, "foundational");
    assert.equal(result.epoch, 12345);
  });
});
