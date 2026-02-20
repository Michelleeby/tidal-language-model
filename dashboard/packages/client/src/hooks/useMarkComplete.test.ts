import { describe, it, expect, vi, beforeEach } from "vitest";

// ---------------------------------------------------------------------------
// Mocks — must be declared before imports that use the mocked modules
// ---------------------------------------------------------------------------

const mockUseMutation = vi.fn();
const mockInvalidateQueries = vi.fn();

vi.mock("@tanstack/react-query", () => ({
  useMutation: (...args: any[]) => mockUseMutation(...args),
  useQueryClient: () => ({ invalidateQueries: mockInvalidateQueries }),
}));

vi.mock("../api/client.js", () => ({
  api: {
    markComplete: vi.fn().mockResolvedValue({
      expId: "exp-1",
      status: { status: "completed", end_time: 1000 },
    }),
  },
}));

import { useMarkComplete } from "./useMarkComplete.js";
import { api } from "../api/client.js";

// ---------------------------------------------------------------------------
// useMarkComplete
// ---------------------------------------------------------------------------

describe("useMarkComplete", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseMutation.mockReturnValue({ mutate: vi.fn(), isPending: false });
  });

  it("mutationFn calls api.markComplete with expId", () => {
    useMarkComplete();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.mutationFn("exp-1");
    expect(api.markComplete).toHaveBeenCalledWith("exp-1");
  });

  it("onSuccess invalidates status and experiments queries", () => {
    useMarkComplete();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.onSuccess(
      { expId: "exp-1", status: { status: "completed" } },
      "exp-1",
    );
    expect(mockInvalidateQueries).toHaveBeenCalledWith({
      queryKey: ["status", "exp-1"],
    });
    expect(mockInvalidateQueries).toHaveBeenCalledWith({
      queryKey: ["experiments"],
    });
  });

  it("invalidates queries using the expId variable, not the response", () => {
    useMarkComplete();
    const callArgs = mockUseMutation.mock.calls[0][0];
    // Even if response has a different expId, use the variable (input)
    callArgs.onSuccess(
      { expId: "response-id", status: { status: "completed" } },
      "input-id",
    );
    expect(mockInvalidateQueries).toHaveBeenCalledWith({
      queryKey: ["status", "input-id"],
    });
  });

  it("returns the mutation result from useMutation", () => {
    const mockMutationResult = { mutate: vi.fn(), isPending: true };
    mockUseMutation.mockReturnValue(mockMutationResult);

    const result = useMarkComplete();
    expect(result).toBe(mockMutationResult);
  });
});
