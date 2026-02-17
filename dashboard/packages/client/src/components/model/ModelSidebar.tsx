import { useExperimentStore } from "../../stores/experimentStore.js";
import { useModelFileTree } from "../../hooks/useModelSource.js";
import FileTree from "./FileTree.js";

export default function ModelSidebar() {
  const {
    selectedFilePath,
    setSelectedFilePath,
    sidebarOpen,
    setSidebarOpen,
  } = useExperimentStore();

  const { data: treeData, isLoading } = useModelFileTree();
  const files = treeData?.files ?? [];

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
          <h2 className="text-sm font-semibold text-gray-200">Tidal Model</h2>
          <button
            className="md:hidden text-gray-400 hover:text-gray-200"
            onClick={() => setSidebarOpen(false)}
          >
            <svg
              className="w-5 h-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* File tree */}
        <div className="flex-1 overflow-y-auto">
          {isLoading ? (
            <div className="px-4 py-3 text-sm text-gray-500">Loading...</div>
          ) : files.length === 0 ? (
            <div className="px-4 py-3 text-sm text-gray-500">
              No source files found.
            </div>
          ) : (
            <FileTree
              files={files}
              selectedPath={selectedFilePath}
              onSelect={(path) => {
                setSelectedFilePath(path);
                setSidebarOpen(false);
              }}
            />
          )}
        </div>
      </aside>
    </>
  );
}
