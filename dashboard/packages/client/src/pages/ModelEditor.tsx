import Editor from "@monaco-editor/react";
import { useExperimentStore } from "../stores/experimentStore.js";
import { useModelFile } from "../hooks/useModelSource.js";

const EXT_TO_LANGUAGE: Record<string, string> = {
  ".py": "python",
  ".yaml": "yaml",
  ".yml": "yaml",
  ".json": "json",
  ".md": "markdown",
  ".txt": "plaintext",
  ".cfg": "ini",
};

function getLanguage(filePath: string): string {
  const dot = filePath.lastIndexOf(".");
  if (dot === -1) return "plaintext";
  return EXT_TO_LANGUAGE[filePath.slice(dot)] ?? "plaintext";
}

export default function ModelEditor() {
  const { selectedFilePath } = useExperimentStore();
  const { data, isLoading } = useModelFile(selectedFilePath);

  if (!selectedFilePath) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Select a file to view
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Loading file...
      </div>
    );
  }

  const language = getLanguage(selectedFilePath);

  return (
    <div className="flex flex-col h-[calc(100vh-64px)]">
      {/* Toolbar */}
      <div className="flex items-center px-4 py-2 border-b border-gray-800">
        <span className="text-sm text-gray-300 font-mono truncate">
          {selectedFilePath}
        </span>
      </div>

      {/* Monaco Editor (read-only) */}
      <div className="flex-1 min-h-0">
        <Editor
          language={language}
          value={data?.content ?? ""}
          theme="vs-dark"
          options={{
            readOnly: true,
            fontSize: 14,
            minimap: { enabled: false },
            wordWrap: "on",
            scrollBeyondLastLine: false,
            automaticLayout: true,
            tabSize: 4,
            insertSpaces: true,
          }}
        />
      </div>
    </div>
  );
}
