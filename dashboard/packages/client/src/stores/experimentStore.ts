import { create } from "zustand";

export type ViewType = "experiments" | "reports";

interface ExperimentStore {
  view: ViewType;
  selectedExpId: string | null;
  comparisonExpIds: string[];
  sidebarOpen: boolean;
  selectedReportId: string | null;
  setView: (view: ViewType) => void;
  setSelectedExpId: (id: string | null) => void;
  toggleComparisonExp: (id: string) => void;
  setSidebarOpen: (open: boolean) => void;
  setSelectedReportId: (id: string | null) => void;
}

export const useExperimentStore = create<ExperimentStore>((set) => ({
  view: "experiments",
  selectedExpId: null,
  comparisonExpIds: [],
  sidebarOpen: true,
  selectedReportId: null,
  setView: (view) => set({ view }),
  setSelectedExpId: (id) => set({ selectedExpId: id }),
  toggleComparisonExp: (id) =>
    set((state) => {
      const ids = state.comparisonExpIds.includes(id)
        ? state.comparisonExpIds.filter((x) => x !== id)
        : [...state.comparisonExpIds, id];
      return { comparisonExpIds: ids };
    }),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  setSelectedReportId: (id) => set({ selectedReportId: id }),
}));
