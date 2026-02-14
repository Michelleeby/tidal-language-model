import { useState } from "react";
import { useConfigFiles, useConfigFile } from "../../hooks/useConfigs.js";

/** Simple YAML syntax highlighting via regex tokenization. */
function highlightYAML(content: string): React.ReactNode[] {
  return content.split("\n").map((line, i) => {
    // Comment lines
    if (/^\s*#/.test(line)) {
      return (
        <div key={i} className="text-gray-500">
          {line}
        </div>
      );
    }

    // Key-value lines
    const kvMatch = line.match(/^(\s*)([\w.-]+)(\s*:\s*)(.*)/);
    if (kvMatch) {
      const [, indent, key, colon, value] = kvMatch;
      return (
        <div key={i}>
          <span>{indent}</span>
          <span className="text-blue-400">{key}</span>
          <span className="text-gray-500">{colon}</span>
          <span className={valueColor(value)}>{value}</span>
        </div>
      );
    }

    // List items
    const listMatch = line.match(/^(\s*-\s)(.*)/);
    if (listMatch) {
      const [, dash, value] = listMatch;
      return (
        <div key={i}>
          <span className="text-gray-500">{dash}</span>
          <span className="text-gray-200">{value}</span>
        </div>
      );
    }

    return (
      <div key={i} className="text-gray-300">
        {line}
      </div>
    );
  });
}

function valueColor(value: string): string {
  const trimmed = value.trim();
  if (trimmed === "true" || trimmed === "false") return "text-orange-400";
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) return "text-green-400";
  if (/^["']/.test(trimmed)) return "text-yellow-300";
  return "text-gray-200";
}

export default function ConfigViewer() {
  const { data: fileList, isLoading: listLoading } = useConfigFiles();
  const files = fileList?.files ?? [];
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  // Auto-select first file
  const activeFile = selectedFile ?? files[0] ?? null;

  const { data: configData, isLoading: fileLoading } =
    useConfigFile(activeFile);

  if (listLoading) {
    return <div className="text-sm text-gray-500">Loading configs...</div>;
  }

  if (files.length === 0) {
    return (
      <div className="text-sm text-gray-500">No config files available</div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Tab switcher */}
      <div className="flex gap-1">
        {files.map((file) => (
          <button
            key={file}
            onClick={() => setSelectedFile(file)}
            className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${
              activeFile === file
                ? "bg-gray-700 text-gray-100"
                : "text-gray-500 hover:text-gray-300 hover:bg-gray-800"
            }`}
          >
            {file}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
        {fileLoading ? (
          <div className="text-sm text-gray-500">Loading...</div>
        ) : configData?.content ? (
          <pre className="font-mono text-xs leading-relaxed">
            {highlightYAML(configData.content)}
          </pre>
        ) : (
          <div className="text-sm text-gray-500">
            Could not load config file
          </div>
        )}
      </div>
    </div>
  );
}
