import { describe, it, expect } from "vitest";
import { HistoryManager } from "../historyManager.js";

// ---------------------------------------------------------------------------
// HistoryManager tests (memento pattern, block-structure undo/redo)
// ---------------------------------------------------------------------------

describe("HistoryManager.push()", () => {
  it("adds to undo stack and enables canUndo()", () => {
    const hm = new HistoryManager<string>();
    expect(hm.canUndo()).toBe(false);

    hm.push("state-1");
    hm.push("state-2");

    expect(hm.canUndo()).toBe(true);
  });

  it("clears redo stack on push", () => {
    const hm = new HistoryManager<string>();
    hm.push("state-1");
    hm.push("state-2");

    // Build up redo stack
    hm.undo("state-2");
    expect(hm.canRedo()).toBe(true);

    // Push new state — redo should be cleared
    hm.push("state-3");
    expect(hm.canRedo()).toBe(false);
  });

  it("respects max depth (drops oldest when full)", () => {
    const hm = new HistoryManager<number>(3); // max 3 undo levels

    hm.push(1);
    hm.push(2);
    hm.push(3);
    hm.push(4); // Oldest (1) is dropped; undoStack = [2, 3, 4]

    // After 4 pushes with max=3, can undo 3 times returning 4, 3, 2.
    // Snapshot 1 (oldest) was dropped and is no longer accessible.
    expect(hm.undo(5)).toBe(4); // current=5 (state after all pushes), returns 4
    expect(hm.undo(4)).toBe(3); // current=4, returns 3
    expect(hm.undo(3)).toBe(2); // current=3, returns 2 (oldest retained)
    expect(hm.undo(2)).toBeNull(); // stack empty — can't undo further
  });
});

describe("HistoryManager.undo()", () => {
  it("returns previous snapshot and enables redo", () => {
    const hm = new HistoryManager<string>();
    hm.push("state-1");
    hm.push("state-2");

    expect(hm.canRedo()).toBe(false);
    const prev = hm.undo("state-3"); // current is state-3
    expect(prev).toBe("state-2");
    expect(hm.canRedo()).toBe(true);
  });

  it("returns null when undo stack is empty", () => {
    const hm = new HistoryManager<string>();
    const result = hm.undo("current");
    expect(result).toBeNull();
  });

  it("multiple undos work sequentially", () => {
    const hm = new HistoryManager<string>();
    hm.push("a");
    hm.push("b");
    hm.push("c");

    expect(hm.undo("d")).toBe("c");
    expect(hm.undo("c")).toBe("b");
    expect(hm.undo("b")).toBe("a");
    expect(hm.undo("a")).toBeNull();
  });
});

describe("HistoryManager.redo()", () => {
  it("returns next snapshot and enables undo", () => {
    const hm = new HistoryManager<string>();
    hm.push("state-1");
    hm.push("state-2");

    hm.undo("state-3"); // current = state-3, undo pops state-2
    expect(hm.canRedo()).toBe(true);

    const next = hm.redo("state-2");
    expect(next).toBe("state-3");
    expect(hm.canUndo()).toBe(true);
  });

  it("returns null when redo stack is empty", () => {
    const hm = new HistoryManager<string>();
    hm.push("state-1");

    const result = hm.redo("state-1");
    expect(result).toBeNull();
  });

  it("undo then redo returns to original state", () => {
    const hm = new HistoryManager<string>();
    hm.push("v1");
    hm.push("v2");

    const afterUndo = hm.undo("v3"); // returns v2, current becomes v2
    expect(afterUndo).toBe("v2");

    const afterRedo = hm.redo("v2"); // returns v3, current becomes v3
    expect(afterRedo).toBe("v3");
  });
});

describe("HistoryManager.clear()", () => {
  it("empties both undo and redo stacks", () => {
    const hm = new HistoryManager<string>();
    hm.push("a");
    hm.push("b");
    hm.undo("c"); // build redo stack

    expect(hm.canUndo()).toBe(true);
    expect(hm.canRedo()).toBe(true);

    hm.clear();

    expect(hm.canUndo()).toBe(false);
    expect(hm.canRedo()).toBe(false);
  });
});
