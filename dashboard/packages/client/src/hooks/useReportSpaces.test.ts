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
    saveReport: vi.fn().mockResolvedValue({ report: { id: "r-1", title: "Saved" } }),
    getReportVersions: vi.fn().mockResolvedValue({ versions: [] }),
    restoreReportVersion: vi.fn().mockResolvedValue({ report: { id: "r-1", title: "Restored" } }),
  },
}));

import { useSaveReport, useReportVersions, useRestoreReportVersion } from "./useReportSpaces.js";
import { api } from "../api/client.js";

// ---------------------------------------------------------------------------
// useSaveReport
// ---------------------------------------------------------------------------

describe("useSaveReport", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseMutation.mockReturnValue({ mutate: vi.fn(), isPending: false });
  });

  it("mutationFn calls api.saveReport with id and body", () => {
    useSaveReport();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.mutationFn({ id: "r-1", title: "New Title", blocks: [] });
    expect(api.saveReport).toHaveBeenCalledWith("r-1", { title: "New Title", blocks: [] });
  });

  it("onSuccess invalidates report, reports, and reportVersions queries", () => {
    useSaveReport();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.onSuccess({ report: { id: "r-1" } }, { id: "r-1", title: "X" });
    expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ["reports", "r-1"] });
    expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ["reports"] });
    expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ["reportVersions", "r-1"] });
  });

  it("returns the mutation result from useMutation", () => {
    const mockResult = { mutate: vi.fn(), isPending: true };
    mockUseMutation.mockReturnValue(mockResult);
    const result = useSaveReport();
    expect(result).toBe(mockResult);
  });
});

// ---------------------------------------------------------------------------
// useReportVersions
// ---------------------------------------------------------------------------

describe("useReportVersions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseQuery.mockReturnValue({ data: undefined, isLoading: false });
  });

  it("passes correct query key with reportId", () => {
    useReportVersions("r-1");
    expect(mockUseQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        queryKey: ["reportVersions", "r-1"],
        enabled: true,
      }),
    );
  });

  it("disables query when id is null", () => {
    useReportVersions(null);
    expect(mockUseQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        enabled: false,
      }),
    );
  });

  it("queryFn calls api.getReportVersions", () => {
    useReportVersions("r-2");
    const callArgs = mockUseQuery.mock.calls[0][0];
    callArgs.queryFn();
    expect(api.getReportVersions).toHaveBeenCalledWith("r-2");
  });

  it("queryFn throws when id is null (defensive guard)", () => {
    useReportVersions(null);
    const callArgs = mockUseQuery.mock.calls[0][0];
    // Even though enabled:false prevents react-query from calling queryFn,
    // a defensive guard should throw if called with null id
    expect(() => callArgs.queryFn()).toThrow();
  });
});

// ---------------------------------------------------------------------------
// useRestoreReportVersion
// ---------------------------------------------------------------------------

describe("useRestoreReportVersion", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseMutation.mockReturnValue({ mutate: vi.fn(), isPending: false });
  });

  it("mutationFn calls api.restoreReportVersion with id and timestamp", () => {
    useRestoreReportVersion();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.mutationFn({ id: "r-1", timestamp: 1700000000000 });
    expect(api.restoreReportVersion).toHaveBeenCalledWith("r-1", 1700000000000);
  });

  it("onSuccess invalidates report, reports, and reportVersions queries", () => {
    useRestoreReportVersion();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.onSuccess({ report: { id: "r-1" } }, { id: "r-1", timestamp: 123 });
    expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ["reports", "r-1"] });
    expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ["reports"] });
    expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ["reportVersions", "r-1"] });
  });

  it("returns the mutation result from useMutation", () => {
    const mockResult = { mutate: vi.fn(), isPending: false };
    mockUseMutation.mockReturnValue(mockResult);
    const result = useRestoreReportVersion();
    expect(result).toBe(mockResult);
  });
});
