import { useState } from "react";
import DashboardPage from "./pages/DashboardPage.js";
import PlaygroundPage from "./pages/PlaygroundPage.js";
import AuthPrompt from "./components/AuthPrompt.js";

type Page = "dashboard" | "playground";

export default function App() {
  const [page, setPage] = useState<Page>("dashboard");

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 overflow-x-hidden">
      <AuthPrompt />
      <header className="border-b border-gray-800 px-3 md:px-6 py-3 flex items-center gap-3 md:gap-6">
        <h1 className="text-lg font-semibold tracking-tight">
          Tidal Dashboard
        </h1>
        <nav className="flex gap-1">
          <NavButton
            active={page === "dashboard"}
            onClick={() => setPage("dashboard")}
          >
            Training
          </NavButton>
          <NavButton
            active={page === "playground"}
            onClick={() => setPage("playground")}
          >
            Playground
          </NavButton>
        </nav>
      </header>
      <main className="p-3 md:p-6">
        {page === "dashboard" && <DashboardPage />}
        {page === "playground" && <PlaygroundPage />}
      </main>
    </div>
  );
}

function NavButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
        active
          ? "bg-blue-600 text-white"
          : "text-gray-400 hover:text-gray-200 hover:bg-gray-800"
      }`}
    >
      {children}
    </button>
  );
}
