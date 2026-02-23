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
    deleteExperiment: vi.fn().mockResolvedValue({ diskDeleted: true, redisKeysRemoved: 0, analysesRemoved: 0 }),
    archiveExperiment: vi.fn().mockResolvedValue({ expId: "exp-1", state: "complete" }),
  },
}));

import { useDeleteExperiment, useArchiveExperiment } from "./useExperimentActions.js";
import { api } from "../api/client.js";

// ---------------------------------------------------------------------------
// useDeleteExperiment
// ---------------------------------------------------------------------------

describe("useDeleteExperiment", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseMutation.mockReturnValue({ mutate: vi.fn(), isPending: false });
  });

  it("mutationFn calls api.deleteExperiment with expId", () => {
    useDeleteExperiment();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.mutationFn("exp-1");
    expect(api.deleteExperiment).toHaveBeenCalledWith("exp-1");
  });

  it("onSuccess invalidates experiments query", () => {
    useDeleteExperiment();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.onSuccess({ diskDeleted: true, redisKeysRemoved: 0, analysesRemoved: 0 }, "exp-1");
    expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ["experiments"] });
  });

  it("returns the mutation result from useMutation", () => {
    const mockResult = { mutate: vi.fn(), isPending: true };
    mockUseMutation.mockReturnValue(mockResult);
    const result = useDeleteExperiment();
    expect(result).toBe(mockResult);
  });
});

// ---------------------------------------------------------------------------
// useArchiveExperiment
// ---------------------------------------------------------------------------

describe("useArchiveExperiment", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseMutation.mockReturnValue({ mutate: vi.fn(), isPending: false });
  });

  it("mutationFn calls api.archiveExperiment with expId", () => {
    useArchiveExperiment();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.mutationFn("exp-2");
    expect(api.archiveExperiment).toHaveBeenCalledWith("exp-2");
  });

  it("onSuccess invalidates experiments query", () => {
    useArchiveExperiment();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.onSuccess({ expId: "exp-2", state: "complete" }, "exp-2");
    expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ["experiments"] });
  });

  it("returns the mutation result from useMutation", () => {
    const mockResult = { mutate: vi.fn(), isPending: false };
    mockUseMutation.mockReturnValue(mockResult);
    const result = useArchiveExperiment();
    expect(result).toBe(mockResult);
  });
});
