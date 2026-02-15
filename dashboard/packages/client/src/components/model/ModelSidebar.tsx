import { useState } from "react";
import { useExperimentStore } from "../../stores/experimentStore.js";
import {
  useUserPlugins,
  useCreateUserPlugin,
  useDeleteUserPlugin,
} from "../../hooks/useUserPlugins.js";
import { usePluginFileTree } from "../../hooks/usePluginFiles.js";
import FileTree from "./FileTree.js";

export default function ModelSidebar() {
  const {
    selectedPluginId,
    setSelectedPluginId,
    selectedFilePath,
    setSelectedFilePath,
    sidebarOpen,
    setSidebarOpen,
  } = useExperimentStore();

  const { data, isLoading } = useUserPlugins();
  const createPlugin = useCreateUserPlugin();
  const deletePlugin = useDeleteUserPlugin();
  const { data: treeData } = usePluginFileTree(selectedPluginId);

  const plugins = data?.plugins ?? [];
  const files = treeData?.files ?? [];

  const [showNewForm, setShowNewForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDisplayName, setNewDisplayName] = useState("");

  const handleCreate = async () => {
    if (!newName.trim() || !newDisplayName.trim()) return;
    try {
      const result = await createPlugin.mutateAsync({
        name: newName.trim(),
        displayName: newDisplayName.trim(),
      });
      setSelectedPluginId(result.plugin.id);
      setShowNewForm(false);
      setNewName("");
      setNewDisplayName("");
      setSidebarOpen(false);
    } catch {
      // Error handling via React Query
    }
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedPluginId === id) {
      setSelectedPluginId(null);
    }
    await deletePlugin.mutateAsync(id);
  };

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
          <h2 className="text-sm font-semibold text-gray-200">Models</h2>
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

        {/* New plugin button / form */}
        <div className="px-3 py-2 border-b border-gray-800">
          {showNewForm ? (
            <div className="space-y-2">
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="plugin-name"
                className="w-full px-2 py-1.5 text-sm bg-gray-900 border border-gray-700 rounded text-gray-200 placeholder-gray-500 outline-none focus:border-blue-500"
              />
              <input
                type="text"
                value={newDisplayName}
                onChange={(e) => setNewDisplayName(e.target.value)}
                placeholder="Display Name"
                className="w-full px-2 py-1.5 text-sm bg-gray-900 border border-gray-700 rounded text-gray-200 placeholder-gray-500 outline-none focus:border-blue-500"
              />
              <div className="flex gap-2">
                <button
                  onClick={handleCreate}
                  disabled={createPlugin.isPending}
                  className="flex-1 px-2 py-1.5 text-sm rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
                >
                  {createPlugin.isPending ? "Creating..." : "Create"}
                </button>
                <button
                  onClick={() => setShowNewForm(false)}
                  className="px-2 py-1.5 text-sm rounded bg-gray-800 text-gray-300 hover:bg-gray-700"
                >
                  Cancel
                </button>
              </div>
              {createPlugin.isError && (
                <div className="text-xs text-red-400">
                  {createPlugin.error.message}
                </div>
              )}
            </div>
          ) : (
            <button
              onClick={() => setShowNewForm(true)}
              className="w-full flex items-center gap-2 px-3 py-2 text-sm rounded transition-colors bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white border border-gray-700"
            >
              <svg
                className="w-4 h-4 flex-shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 4.5v15m7.5-7.5h-15"
                />
              </svg>
              New Plugin
            </button>
          )}
        </div>

        {/* Plugin list + file tree */}
        <div className="flex-1 overflow-y-auto">
          {isLoading ? (
            <div className="px-4 py-3 text-sm text-gray-500">Loading...</div>
          ) : plugins.length === 0 ? (
            <div className="px-4 py-3 text-sm text-gray-500">
              No plugins yet. Create one to get started.
            </div>
          ) : (
            <ul className="py-1">
              {plugins.map((plugin) => (
                <li key={plugin.id}>
                  <button
                    onClick={() => {
                      setSelectedPluginId(plugin.id);
                      setSidebarOpen(false);
                    }}
                    className={`w-full text-left px-4 py-2.5 transition-colors group ${
                      selectedPluginId === plugin.id
                        ? "bg-gray-800 border-l-2 border-blue-500"
                        : "hover:bg-gray-900 border-l-2 border-transparent"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-200 truncate">
                        {plugin.displayName}
                      </span>
                      <button
                        onClick={(e) => handleDelete(plugin.id, e)}
                        className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-400 transition-opacity p-0.5"
                        title="Delete plugin"
                      >
                        <svg
                          className="w-3.5 h-3.5"
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
                    <div className="mt-0.5 text-xs text-gray-500 font-mono">
                      {plugin.name}
                    </div>
                  </button>

                  {/* Show file tree for selected plugin */}
                  {selectedPluginId === plugin.id && (
                    <div className="border-l-2 border-blue-500/30 ml-4">
                      <FileTree
                        files={files}
                        selectedPath={selectedFilePath}
                        onSelect={setSelectedFilePath}
                      />
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>
    </>
  );
}
