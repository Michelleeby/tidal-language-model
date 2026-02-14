import { create } from "zustand";

interface ExperimentStore {
  selectedExpId: string | null;
  comparisonExpIds: string[];
  sidebarOpen: boolean;
  setSelectedExpId: (id: string | null) => void;
  toggleComparisonExp: (id: string) => void;
  setSidebarOpen: (open: boolean) => void;
}

export const useExperimentStore = create<ExperimentStore>((set) => ({
  selectedExpId: null,
  comparisonExpIds: [],
  sidebarOpen: true,
  setSelectedExpId: (id) => set({ selectedExpId: id }),
  toggleComparisonExp: (id) =>
    set((state) => {
      const ids = state.comparisonExpIds.includes(id)
        ? state.comparisonExpIds.filter((x) => x !== id)
        : [...state.comparisonExpIds, id];
      return { comparisonExpIds: ids };
    }),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
}));
