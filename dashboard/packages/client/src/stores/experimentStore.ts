import { create } from "zustand";

type Tab = "training" | "rl-gating" | "comparison" | "checkpoints" | "samples" | "logs";

interface ExperimentStore {
  selectedExpId: string | null;
  comparisonExpIds: string[];
  activeTab: Tab;
  setSelectedExpId: (id: string | null) => void;
  toggleComparisonExp: (id: string) => void;
  setActiveTab: (tab: Tab) => void;
}

export const useExperimentStore = create<ExperimentStore>((set) => ({
  selectedExpId: null,
  comparisonExpIds: [],
  activeTab: "training",
  setSelectedExpId: (id) => set({ selectedExpId: id }),
  toggleComparisonExp: (id) =>
    set((state) => {
      const ids = state.comparisonExpIds.includes(id)
        ? state.comparisonExpIds.filter((x) => x !== id)
        : [...state.comparisonExpIds, id];
      return { comparisonExpIds: ids };
    }),
  setActiveTab: (tab) => set({ activeTab: tab }),
}));
