import { describe, it, expect, vi, beforeEach } from "vitest";

// ---------------------------------------------------------------------------
// Mocks — must be declared before imports that use the mocked modules
// ---------------------------------------------------------------------------

const mockUseQuery = vi.fn();
const mockUseMutation = vi.fn();
const mockInvalidateQueries = vi.fn();

vi.mock("@tanstack/react-query", () => ({
  useQuery: (...args: any[]) => mockUseQuery(...args),
  useMutation: (...args: any[]) => mockUseMutation(...args),
  useQueryClient: () => ({ invalidateQueries: mockInvalidateQueries }),
}));

vi.mock("../api/client.js", () => ({
  api: {
    listAnalyses: vi.fn().mockResolvedValue({ analyses: [] }),
    getAnalysis: vi.fn().mockResolvedValue({ analysis: {} }),
    createAnalysis: vi.fn().mockResolvedValue({ analysis: {} }),
    deleteAnalysis: vi.fn().mockResolvedValue({ ok: true }),
  },
}));

import {
  useAnalyses,
  useAnalysis,
  useSaveAnalysis,
  useDeleteAnalysis,
} from "./useAnalyses.js";
import { api } from "../api/client.js";

// ---------------------------------------------------------------------------
// useAnalyses
// ---------------------------------------------------------------------------

describe("useAnalyses", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseQuery.mockReturnValue({ data: undefined, isLoading: false });
    mockUseMutation.mockReturnValue({ mutate: vi.fn() });
  });

  it("passes correct query key with expId and type", () => {
    useAnalyses("exp-1", "trajectory");
    expect(mockUseQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        queryKey: ["analyses", "exp-1", "trajectory"],
        enabled: true,
      }),
    );
  });

  it("disables query when expId is null", () => {
    useAnalyses(null);
    expect(mockUseQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        enabled: false,
      }),
    );
  });

  it("queryFn calls api.listAnalyses with correct arguments", () => {
    useAnalyses("exp-2", "sweep");
    const callArgs = mockUseQuery.mock.calls[0][0];
    callArgs.queryFn();
    expect(api.listAnalyses).toHaveBeenCalledWith("exp-2", "sweep");
  });

  it("passes type=undefined when not specified", () => {
    useAnalyses("exp-3");
    expect(mockUseQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        queryKey: ["analyses", "exp-3", undefined],
      }),
    );
  });
});

// ---------------------------------------------------------------------------
// useAnalysis
// ---------------------------------------------------------------------------

describe("useAnalysis", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseQuery.mockReturnValue({ data: undefined, isLoading: false });
  });

  it("passes correct query key", () => {
    useAnalysis("analysis-1");
    expect(mockUseQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        queryKey: ["analysis", "analysis-1"],
        enabled: true,
      }),
    );
  });

  it("disables query when id is null", () => {
    useAnalysis(null);
    expect(mockUseQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        enabled: false,
      }),
    );
  });

  it("queryFn calls api.getAnalysis", () => {
    useAnalysis("analysis-2");
    const callArgs = mockUseQuery.mock.calls[0][0];
    callArgs.queryFn();
    expect(api.getAnalysis).toHaveBeenCalledWith("analysis-2");
  });
});

// ---------------------------------------------------------------------------
// useSaveAnalysis
// ---------------------------------------------------------------------------

describe("useSaveAnalysis", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseMutation.mockReturnValue({ mutate: vi.fn() });
  });

  it("mutationFn calls api.createAnalysis with expId and body", () => {
    useSaveAnalysis();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.mutationFn({
      expId: "exp-1",
      analysisType: "trajectory",
      label: "test",
      request: {},
      data: {},
    });
    expect(api.createAnalysis).toHaveBeenCalledWith("exp-1", {
      analysisType: "trajectory",
      label: "test",
      request: {},
      data: {},
    });
  });

  it("onSuccess invalidates analyses queries for the experiment", () => {
    useSaveAnalysis();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.onSuccess({}, { expId: "exp-1" });
    expect(mockInvalidateQueries).toHaveBeenCalledWith({
      queryKey: ["analyses", "exp-1"],
    });
  });
});

// ---------------------------------------------------------------------------
// useDeleteAnalysis
// ---------------------------------------------------------------------------

describe("useDeleteAnalysis", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseMutation.mockReturnValue({ mutate: vi.fn() });
  });

  it("mutationFn calls api.deleteAnalysis", () => {
    useDeleteAnalysis();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.mutationFn("analysis-1");
    expect(api.deleteAnalysis).toHaveBeenCalledWith("analysis-1");
  });

  it("onSuccess invalidates all analyses queries", () => {
    useDeleteAnalysis();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.onSuccess();
    expect(mockInvalidateQueries).toHaveBeenCalledWith({
      queryKey: ["analyses"],
    });
  });
});
