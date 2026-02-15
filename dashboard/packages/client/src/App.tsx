import { useEffect } from "react";
import ExperimentSidebar from "./components/notebook/ExperimentSidebar.js";
import ExperimentNotebook from "./pages/ExperimentNotebook.js";
import ReportsSidebar from "./components/reports/ReportsSidebar.js";
import ReportEditor from "./pages/ReportEditor.js";
import ModelSidebar from "./components/model/ModelSidebar.js";
import ModelEditor from "./pages/ModelEditor.js";
import LoginPage from "./components/LoginPage.js";
import UserMenu from "./components/UserMenu.js";
import { useAuthStore, checkAuth } from "./hooks/useAuth.js";
import { useExperimentStore } from "./stores/experimentStore.js";
import type { ViewType } from "./stores/experimentStore.js";

const TABS: { id: ViewType; label: string }[] = [
  { id: "experiments", label: "Experiments" },
  { id: "reports", label: "Reports" },
  { id: "model", label: "Model" },
];

export default function App() {
  const { view, setView, setSidebarOpen } = useExperimentStore();
  const user = useAuthStore((s) => s.user);
  const loading = useAuthStore((s) => s.loading);

  useEffect(() => {
    checkAuth();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-gray-400 text-sm">Loading...</div>
      </div>
    );
  }

  if (!user) {
    return <LoginPage />;
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 overflow-x-hidden">
      {/* Header */}
      <header className="border-b border-gray-800 px-3 md:px-6 py-3 flex items-center gap-3">
        <button
          className="md:hidden text-gray-400 hover:text-gray-200"
          onClick={() => setSidebarOpen(true)}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        <h1 className="text-lg font-semibold tracking-tight">
          🌊 Tidal
        </h1>

        {/* Nav tabs */}
        <nav className="flex items-center gap-1 ml-6">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setView(tab.id)}
              className={`px-3 py-1 text-sm rounded transition-colors ${
                view === tab.id
                  ? "bg-gray-800 text-gray-100"
                  : "text-gray-400 hover:text-gray-200 hover:bg-gray-900"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <UserMenu />
      </header>

      {/* Sidebar + Main Content */}
      <div className="flex">
        {view === "experiments" ? (
          <>
            <ExperimentSidebar />
            <main className="flex-1 min-w-0 p-3 md:p-6">
              <ExperimentNotebook />
            </main>
          </>
        ) : view === "reports" ? (
          <>
            <ReportsSidebar />
            <main className="flex-1 min-w-0 p-3 md:p-6">
              <ReportEditor />
            </main>
          </>
        ) : (
          <>
            <ModelSidebar />
            <main className="flex-1 min-w-0">
              <ModelEditor />
            </main>
          </>
        )}
      </div>
    </div>
  );
}
