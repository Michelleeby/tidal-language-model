import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// ---------------------------------------------------------------------------
// Mocks — intercept React hooks to test logic without a render context
// ---------------------------------------------------------------------------

let stateValue: any = null;
const mockSetState = vi.fn((v: any) => {
  stateValue = typeof v === "function" ? v(stateValue) : v;
});
let refObject: { current: any };

vi.mock("react", () => ({
  useState: (init: any) => {
    if (stateValue === undefined) stateValue = init;
    return [stateValue, mockSetState];
  },
  useRef: (init: any) => {
    if (!refObject) refObject = { current: init };
    return refObject;
  },
  useCallback: (fn: any) => fn,
}));

import { useSpacesStatus } from "./useSpacesStatus.js";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useSpacesStatus", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    stateValue = null;
    refObject = { current: null };
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("returns null status initially", () => {
    const { spacesStatus } = useSpacesStatus();
    expect(spacesStatus).toBeNull();
  });

  it("showStatus sets a success message", () => {
    const { showStatus } = useSpacesStatus();
    showStatus("success", "Saved to Spaces");
    expect(mockSetState).toHaveBeenCalledWith({ type: "success", message: "Saved to Spaces" });
  });

  it("showStatus sets an error message", () => {
    const { showStatus } = useSpacesStatus();
    showStatus("error", "Save failed");
    expect(mockSetState).toHaveBeenCalledWith({ type: "error", message: "Save failed" });
  });

  it("showStatus stores timer ID in ref", () => {
    const { showStatus } = useSpacesStatus();
    showStatus("success", "Done");
    expect(refObject.current).not.toBeNull();
  });

  it("auto-clears status after 3 seconds", () => {
    const { showStatus } = useSpacesStatus();
    showStatus("success", "Saved");

    vi.advanceTimersByTime(3000);

    expect(mockSetState).toHaveBeenCalledWith(null);
  });

  it("showStatus clears previous timer before setting a new one", () => {
    const clearTimeoutSpy = vi.spyOn(globalThis, "clearTimeout");

    const { showStatus } = useSpacesStatus();
    showStatus("success", "First");
    const firstTimer = refObject.current;

    showStatus("error", "Second");
    expect(clearTimeoutSpy).toHaveBeenCalledWith(firstTimer);

    clearTimeoutSpy.mockRestore();
  });

  it("clearStatus clears timer and resets state", () => {
    const clearTimeoutSpy = vi.spyOn(globalThis, "clearTimeout");

    const { showStatus, clearStatus } = useSpacesStatus();
    showStatus("success", "Saved");
    const timer = refObject.current;

    clearStatus();

    expect(clearTimeoutSpy).toHaveBeenCalledWith(timer);
    expect(refObject.current).toBeNull();
    expect(mockSetState).toHaveBeenCalledWith(null);

    clearTimeoutSpy.mockRestore();
  });
});
