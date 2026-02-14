import ExperimentSidebar from "./components/notebook/ExperimentSidebar.js";
import ExperimentNotebook from "./pages/ExperimentNotebook.js";
import AuthPrompt from "./components/AuthPrompt.js";
import { useExperimentStore } from "./stores/experimentStore.js";

export default function App() {
  const { setSidebarOpen } = useExperimentStore();

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 overflow-x-hidden">
      <AuthPrompt />

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
          Tidal Dashboard
        </h1>
      </header>

      {/* Sidebar + Main Content */}
      <div className="flex">
        <ExperimentSidebar />
        <main className="flex-1 min-w-0 p-3 md:p-6">
          <ExperimentNotebook />
        </main>
      </div>
    </div>
  );
}
