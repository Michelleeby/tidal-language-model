import { useState } from "react";
import { useExperimentStore } from "../../stores/experimentStore.js";
import {
  useUserPlugins,
  useCreateUserPlugin,
  useDeleteUserPlugin,
} from "../../hooks/useUserPlugins.js";
import { usePluginFileTree } from "../../hooks/usePluginFiles.js";
import {
  usePluginGitStatus,
  usePullPlugin,
  usePushPlugin,
} from "../../hooks/usePluginGit.js";
import FileTree from "./FileTree.js";

function GitSyncBar({ pluginId, repoUrl }: { pluginId: string; repoUrl: string }) {
  const { data: gitStatus } = usePluginGitStatus(pluginId);
  const pullPlugin = usePullPlugin();
  const pushPlugin = usePushPlugin();
  const [showPush, setShowPush] = useState(false);
  const [commitMsg, setCommitMsg] = useState("");

  const dirty = gitStatus?.dirty ?? false;
  const fileCount = gitStatus?.files?.length ?? 0;

  const handlePush = () => {
    if (!commitMsg.trim()) return;
    pushPlugin.mutate(
      { pluginId, message: commitMsg.trim() },
      {
        onSuccess: () => {
          setShowPush(false);
          setCommitMsg("");
        },
      },
    );
  };

  return (
    <div className="px-3 py-2 border-t border-gray-800/50 space-y-1.5">
      {/* Status row */}
      <div className="flex items-center gap-2">
        <div
          className={`w-2 h-2 rounded-full flex-shrink-0 ${
            dirty ? "bg-yellow-500" : "bg-green-500"
          }`}
          title={dirty ? `${fileCount} changed file(s)` : "Clean"}
        />
        <span className="text-xs text-gray-400 truncate">
          {dirty ? `${fileCount} changed` : "Clean"}
        </span>

        {/* GitHub link */}
        {repoUrl && (
          <a
            href={repoUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-auto text-gray-500 hover:text-gray-300"
            title="Open on GitHub"
          >
            <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
          </a>
        )}
      </div>

      {/* Action buttons */}
      <div className="flex gap-1.5">
        <button
          onClick={() => pullPlugin.mutate(pluginId)}
          disabled={pullPlugin.isPending}
          className="flex-1 px-2 py-1 text-xs rounded bg-gray-800 text-gray-300 hover:bg-gray-700 disabled:opacity-50 transition-colors"
        >
          {pullPlugin.isPending ? "Pulling..." : "Pull"}
        </button>
        <button
          onClick={() => setShowPush(true)}
          disabled={!dirty || pushPlugin.isPending}
          className="flex-1 px-2 py-1 text-xs rounded bg-gray-800 text-gray-300 hover:bg-gray-700 disabled:opacity-50 transition-colors"
        >
          {pushPlugin.isPending ? "Pushing..." : "Push"}
        </button>
      </div>

      {/* Push commit message input */}
      {showPush && (
        <div className="space-y-1.5">
          <input
            type="text"
            value={commitMsg}
            onChange={(e) => setCommitMsg(e.target.value)}
            placeholder="Commit message"
            className="w-full px-2 py-1 text-xs bg-gray-900 border border-gray-700 rounded text-gray-200 placeholder-gray-500 outline-none focus:border-blue-500"
            onKeyDown={(e) => {
              if (e.key === "Enter") handlePush();
              if (e.key === "Escape") setShowPush(false);
            }}
            autoFocus
          />
          <div className="flex gap-1.5">
            <button
              onClick={handlePush}
              disabled={pushPlugin.isPending || !commitMsg.trim()}
              className="flex-1 px-2 py-1 text-xs rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50 transition-colors"
            >
              {pushPlugin.isPending ? "Pushing..." : "Commit & Push"}
            </button>
            <button
              onClick={() => setShowPush(false)}
              className="px-2 py-1 text-xs rounded bg-gray-800 text-gray-300 hover:bg-gray-700"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

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

  // Find selected plugin for its repoUrl
  const selectedPlugin = plugins.find((p) => p.id === selectedPluginId);

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
                placeholder="plugin_name"
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

                  {/* Show file tree + git sync for selected plugin */}
                  {selectedPluginId === plugin.id && (
                    <div className="border-l-2 border-blue-500/30 ml-4">
                      <FileTree
                        files={files}
                        selectedPath={selectedFilePath}
                        onSelect={setSelectedFilePath}
                      />
                      {plugin.githubRepoUrl && (
                        <GitSyncBar
                          pluginId={plugin.id}
                          repoUrl={plugin.githubRepoUrl}
                        />
                      )}
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
