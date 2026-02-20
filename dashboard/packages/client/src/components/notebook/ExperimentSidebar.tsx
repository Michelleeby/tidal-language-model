import { useExperiments } from "../../hooks/useExperiments.js";
import { useExperimentStore } from "../../stores/experimentStore.js";
import type { ExperimentSummary } from "@tidal/shared";

function statusColor(exp: ExperimentSummary): string {
  if (!exp.status) return "bg-gray-500";
  switch (exp.status.status) {
    case "training":
      return "bg-blue-500 animate-pulse";
    case "completed":
      return "bg-green-500";
    case "initialized":
      return "bg-yellow-500";
    default:
      return "bg-gray-500";
  }
}

interface TypeBadge {
  label: string;
  color: string;
}

function typeBadges(exp: ExperimentSummary): TypeBadge[] {
  const badges: TypeBadge[] = [];

  if (exp.experimentType === "rl") {
    badges.push({ label: "RL", color: "bg-purple-700 text-purple-200" });
  } else if (exp.experimentType === "lm") {
    badges.push({ label: "LM", color: "bg-blue-700 text-blue-200" });
  } else {
    // Legacy/unknown — infer from checkpoints
    const hasFoundational = exp.checkpoints.some((c) =>
      c.startsWith("checkpoint_foundational"),
    );
    const hasRL = exp.checkpoints.some((c) => c.startsWith("rl_checkpoint"));
    if (hasFoundational) badges.push({ label: "LM", color: "bg-blue-700 text-blue-200" });
    if (hasRL) badges.push({ label: "RL", color: "bg-purple-700 text-purple-200" });
  }

  if (exp.hasEvaluation) badges.push({ label: "Eval", color: "bg-gray-700 text-gray-300" });
  return badges;
}

export default function ExperimentSidebar() {
  const { selectedExpId, setSelectedExpId, sidebarOpen, setSidebarOpen } =
    useExperimentStore();
  const { data: expData, isLoading } = useExperiments();

  const experiments = expData?.experiments ?? [];

  return (
    <>
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-20 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside
        className={`
          fixed md:static z-30 top-0 left-0 h-full md:h-auto
          w-64 bg-gray-950 border-r border-gray-800
          flex flex-col transition-transform duration-200
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}
          md:translate-x-0
        `}
      >
        {/* Header */}
        <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-200">Experiments</h2>
          <button
            className="md:hidden text-gray-400 hover:text-gray-200"
            onClick={() => setSidebarOpen(false)}
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* New experiment button */}
        <div className="px-3 py-2 border-b border-gray-800">
          <button
            onClick={() => {
              setSelectedExpId(null);
              setSidebarOpen(false);
            }}
            className={`w-full flex items-center gap-2 px-3 py-2 text-sm rounded transition-colors ${
              selectedExpId === null
                ? "bg-blue-600/20 text-blue-400 border border-blue-500/30"
                : "bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white border border-gray-700"
            }`}
          >
            <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
            </svg>
            New Experiment
          </button>
        </div>

        {/* Experiment list */}
        <div className="flex-1 overflow-y-auto">
          {isLoading ? (
            <div className="px-4 py-3 text-sm text-gray-500">Loading...</div>
          ) : experiments.length === 0 ? (
            <div className="px-4 py-3 text-sm text-gray-500">
              No experiments yet
            </div>
          ) : (
            <ul className="py-1">
              {experiments.map((exp) => (
                <li key={exp.id}>
                  <button
                    onClick={() => {
                      setSelectedExpId(exp.id);
                      setSidebarOpen(false);
                    }}
                    className={`w-full text-left px-4 py-2.5 transition-colors ${
                      selectedExpId === exp.id
                        ? "bg-gray-800 border-l-2 border-blue-500"
                        : "hover:bg-gray-900 border-l-2 border-transparent"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={`w-2 h-2 rounded-full flex-shrink-0 ${statusColor(exp)}`}
                      />
                      <span className="text-sm text-gray-200 truncate font-mono">
                        {exp.id.length > 16 ? `${exp.id.slice(0, 16)}...` : exp.id}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mt-1 ml-4">
                      <span className="text-xs text-gray-500">
                        {new Date(exp.created).toLocaleDateString()}
                      </span>
                      <span className="text-xs text-gray-500">
                        {exp.checkpoints.length} ckpt{exp.checkpoints.length !== 1 ? "s" : ""}
                      </span>
                      {typeBadges(exp).map((badge) => (
                        <span
                          key={badge.label}
                          className={`text-[10px] px-1.5 py-0.5 rounded ${badge.color}`}
                        >
                          {badge.label}
                        </span>
                      ))}
                    </div>
                    {exp.experimentType === "rl" && exp.sourceExperimentId && (
                      <div className="ml-4 mt-0.5">
                        <span className="text-[10px] text-gray-600 truncate block">
                          from {exp.sourceExperimentId.slice(0, 16)}...
                        </span>
                      </div>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>
    </>
  );
}
