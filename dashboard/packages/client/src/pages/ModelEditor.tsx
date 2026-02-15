import { useState, useEffect, useRef, useCallback } from "react";
import Editor from "@monaco-editor/react";
import { useExperimentStore } from "../stores/experimentStore.js";
import { usePluginFile, useSavePluginFile } from "../hooks/usePluginFiles.js";

type SaveStatus = "saved" | "saving" | "unsaved";

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
  const { selectedPluginId, selectedFilePath } = useExperimentStore();
  const { data, isLoading } = usePluginFile(selectedPluginId, selectedFilePath);
  const saveFile = useSavePluginFile();

  const [content, setContent] = useState("");
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("saved");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const contentRef = useRef(content);
  contentRef.current = content;

  // Sync content when file data loads or file changes
  useEffect(() => {
    if (data?.content !== undefined) {
      setContent(data.content);
      setSaveStatus("saved");
    }
  }, [data]);

  const save = useCallback(
    async (text: string) => {
      if (!selectedPluginId || !selectedFilePath) return;
      setSaveStatus("saving");
      try {
        await saveFile.mutateAsync({
          pluginId: selectedPluginId,
          filePath: selectedFilePath,
          content: text,
        });
        setSaveStatus("saved");
      } catch {
        setSaveStatus("unsaved");
      }
    },
    [selectedPluginId, selectedFilePath, saveFile],
  );

  const debouncedSave = useCallback(
    (text: string) => {
      setSaveStatus("unsaved");
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => save(text), 1500);
    },
    [save],
  );

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const handleChange = (value: string | undefined) => {
    const newContent = value ?? "";
    setContent(newContent);
    debouncedSave(newContent);
  };

  if (!selectedPluginId) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Select a plugin or create a new one
      </div>
    );
  }

  if (!selectedFilePath) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Select a file from the sidebar
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
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800">
        <span className="text-sm text-gray-300 font-mono truncate">
          {selectedFilePath}
        </span>
        <span
          className={`text-xs px-2 py-0.5 rounded flex-shrink-0 ${
            saveStatus === "saved"
              ? "text-green-400 bg-green-900/30"
              : saveStatus === "saving"
                ? "text-yellow-400 bg-yellow-900/30"
                : "text-gray-400 bg-gray-800"
          }`}
        >
          {saveStatus === "saved"
            ? "Saved"
            : saveStatus === "saving"
              ? "Saving..."
              : "Unsaved"}
        </span>
      </div>

      {/* Monaco Editor */}
      <div className="flex-1 min-h-0">
        <Editor
          language={language}
          value={content}
          onChange={handleChange}
          theme="vs-dark"
          options={{
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
