import { describe, it, expect, vi, beforeEach } from "vitest";

// ---------------------------------------------------------------------------
// Mocks — must be declared before imports that use the mocked modules
// ---------------------------------------------------------------------------

const mockUseMutation = vi.fn();
const mockUseQuery = vi.fn();
const mockInvalidateQueries = vi.fn();

vi.mock("@tanstack/react-query", () => ({
  useMutation: (...args: any[]) => mockUseMutation(...args),
  useQuery: (...args: any[]) => mockUseQuery(...args),
  useQueryClient: () => ({ invalidateQueries: mockInvalidateQueries }),
}));

const mockSetSelectedExpId = vi.fn();
vi.mock("../stores/experimentStore.js", () => ({
  useExperimentStore: (selector: any) =>
    selector({ setSelectedExpId: mockSetSelectedExpId }),
}));

vi.mock("../api/client.js", () => ({
  api: {
    createJob: vi.fn().mockResolvedValue({
      job: {
        jobId: "job-123",
        experimentId: "exp-456",
        status: "pending",
      },
    }),
    getJobs: vi.fn().mockResolvedValue({ jobs: [] }),
    getActiveJob: vi.fn().mockResolvedValue({ job: null }),
    signalJob: vi.fn(),
    cancelJob: vi.fn(),
  },
}));

import { useCreateJob } from "./useJobs.js";
import { api } from "../api/client.js";

// ---------------------------------------------------------------------------
// useCreateJob
// ---------------------------------------------------------------------------

describe("useCreateJob", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseMutation.mockReturnValue({ mutate: vi.fn(), isPending: false });
  });

  it("mutationFn calls api.createJob with request body", () => {
    useCreateJob();
    const callArgs = mockUseMutation.mock.calls[0][0];
    const body = {
      type: "rl-training",
      plugin: "tidal",
      configPath: "configs/base_config.yaml",
    };
    callArgs.mutationFn(body);
    expect(api.createJob).toHaveBeenCalledWith(body);
  });

  it("onSuccess sets selectedExpId from response experimentId", () => {
    useCreateJob();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.onSuccess({
      job: {
        jobId: "job-123",
        experimentId: "exp-new-789",
        status: "pending",
      },
    });
    expect(mockSetSelectedExpId).toHaveBeenCalledWith("exp-new-789");
  });

  it("onSuccess invalidates experiments query for sidebar refresh", () => {
    useCreateJob();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.onSuccess({
      job: {
        jobId: "job-123",
        experimentId: "exp-new-789",
        status: "pending",
      },
    });
    expect(mockInvalidateQueries).toHaveBeenCalledWith({
      queryKey: ["experiments"],
    });
  });

  it("onSuccess invalidates jobs query", () => {
    useCreateJob();
    const callArgs = mockUseMutation.mock.calls[0][0];
    callArgs.onSuccess({
      job: {
        jobId: "job-123",
        experimentId: "exp-new-789",
        status: "pending",
      },
    });
    expect(mockInvalidateQueries).toHaveBeenCalledWith({
      queryKey: ["jobs"],
    });
  });

  it("onSuccess handles missing experimentId gracefully", () => {
    useCreateJob();
    const callArgs = mockUseMutation.mock.calls[0][0];
    // Job without experimentId (shouldn't happen, but belt-and-suspenders)
    callArgs.onSuccess({
      job: {
        jobId: "job-123",
        status: "pending",
      },
    });
    expect(mockSetSelectedExpId).not.toHaveBeenCalled();
  });
});
