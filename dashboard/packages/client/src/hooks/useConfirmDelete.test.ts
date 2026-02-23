import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// ---------------------------------------------------------------------------
// Mocks — intercept React hooks to test logic without a render context
// ---------------------------------------------------------------------------

let stateValue: string | null = null;
const mockSetState = vi.fn((v: any) => {
  stateValue = typeof v === "function" ? v(stateValue) : v;
});
let refObject: { current: any };
let effectCleanup: (() => void) | null = null;

vi.mock("react", () => ({
  useState: (init: any) => {
    if (stateValue === undefined) stateValue = init;
    return [stateValue, mockSetState];
  },
  useRef: (init: any) => {
    if (!refObject) refObject = { current: init };
    return refObject;
  },
  useEffect: (effect: () => (() => void) | void) => {
    const cleanup = effect();
    if (typeof cleanup === "function") {
      effectCleanup = cleanup;
    }
  },
  useCallback: (fn: any) => fn,
}));

import { useConfirmDelete } from "./useConfirmDelete.js";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useConfirmDelete", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    stateValue = null;
    refObject = { current: null };
    effectCleanup = null;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("returns confirmDeleteId as null initially", () => {
    const { confirmDeleteId } = useConfirmDelete();
    expect(confirmDeleteId).toBeNull();
  });

  it("requestConfirm sets the confirmDeleteId", () => {
    const { requestConfirm } = useConfirmDelete();
    requestConfirm("exp-1");
    expect(mockSetState).toHaveBeenCalledWith("exp-1");
  });

  it("requestConfirm stores the timer ID in the ref", () => {
    const { requestConfirm } = useConfirmDelete();
    requestConfirm("exp-1");
    expect(refObject.current).not.toBeNull();
  });

  it("requestConfirm clears previous timer before setting a new one", () => {
    const clearTimeoutSpy = vi.spyOn(globalThis, "clearTimeout");

    const { requestConfirm } = useConfirmDelete();

    // First call — sets a timer
    requestConfirm("exp-1");
    const firstTimer = refObject.current;
    expect(firstTimer).not.toBeNull();

    // Second call — should clear the first timer
    requestConfirm("exp-2");
    expect(clearTimeoutSpy).toHaveBeenCalledWith(firstTimer);

    clearTimeoutSpy.mockRestore();
  });

  it("auto-resets confirmDeleteId after 3 seconds", () => {
    const { requestConfirm } = useConfirmDelete();
    requestConfirm("exp-1");

    // Advance time past 3s
    vi.advanceTimersByTime(3000);

    // The setTimeout callback should have called setConfirmDeleteId(null)
    expect(mockSetState).toHaveBeenCalledWith(null);
  });

  it("clearConfirm clears the timer and resets state", () => {
    const clearTimeoutSpy = vi.spyOn(globalThis, "clearTimeout");

    const { requestConfirm, clearConfirm } = useConfirmDelete();
    requestConfirm("exp-1");
    const timer = refObject.current;

    clearConfirm();

    expect(clearTimeoutSpy).toHaveBeenCalledWith(timer);
    expect(refObject.current).toBeNull();
    expect(mockSetState).toHaveBeenCalledWith(null);

    clearTimeoutSpy.mockRestore();
  });

  it("useEffect cleanup clears the timer on unmount", () => {
    const clearTimeoutSpy = vi.spyOn(globalThis, "clearTimeout");

    const { requestConfirm } = useConfirmDelete();
    requestConfirm("exp-1");
    const timer = refObject.current;

    // Simulate unmount
    expect(effectCleanup).toBeTypeOf("function");
    effectCleanup!();

    expect(clearTimeoutSpy).toHaveBeenCalledWith(timer);

    clearTimeoutSpy.mockRestore();
  });

  it("useEffect cleanup is safe when no timer is pending", () => {
    useConfirmDelete();

    // No timer set — cleanup should not throw
    expect(effectCleanup).toBeTypeOf("function");
    expect(() => effectCleanup!()).not.toThrow();
  });
});
