/**
 * Memento pattern: session-only undo/redo for block-structure changes.
 *
 * Operates at the block level (add/remove/reorder blocks, cross-block content
 * changes). BlockNote handles text-level undo within a paragraph.
 *
 * Stacks are in-memory only — cleared on page navigation.
 */
export class HistoryManager<T> {
  private undoStack: T[] = [];
  private redoStack: T[] = [];

  constructor(private maxDepth = 50) {}

  /**
   * Record a new snapshot.
   * Pushes the current state onto the undo stack and clears the redo stack.
   * Drops the oldest entry if the stack exceeds maxDepth.
   */
  push(snapshot: T): void {
    this.undoStack.push(snapshot);
    if (this.undoStack.length > this.maxDepth) {
      this.undoStack.shift(); // Drop oldest
    }
    this.redoStack = [];
  }

  /**
   * Undo: returns the previous snapshot (top of undo stack) and pushes
   * `current` onto the redo stack. Returns null if nothing to undo.
   */
  undo(current: T): T | null {
    if (this.undoStack.length === 0) return null;
    const prev = this.undoStack.pop()!;
    this.redoStack.push(current);
    return prev;
  }

  /**
   * Redo: returns the next snapshot (top of redo stack) and pushes
   * `current` onto the undo stack. Returns null if nothing to redo.
   */
  redo(current: T): T | null {
    if (this.redoStack.length === 0) return null;
    const next = this.redoStack.pop()!;
    this.undoStack.push(current);
    return next;
  }

  canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  /** Clear both stacks (e.g. on page navigation). */
  clear(): void {
    this.undoStack = [];
    this.redoStack = [];
  }
}
