import { useState } from "react";
import type { PluginFileNode } from "@tidal/shared";

interface FileTreeNodeProps {
  node: PluginFileNode;
  selectedPath: string | null;
  onSelect: (path: string) => void;
  depth?: number;
}

const EXT_ICONS: Record<string, string> = {
  ".py": "Py",
  ".yaml": "Ya",
  ".yml": "Ya",
  ".json": "Js",
  ".md": "Md",
  ".txt": "Tx",
  ".cfg": "Cf",
};

function getFileIcon(name: string): string {
  const dot = name.lastIndexOf(".");
  if (dot === -1) return "Fi";
  return EXT_ICONS[name.slice(dot)] ?? "Fi";
}

export default function FileTreeNode({
  node,
  selectedPath,
  onSelect,
  depth = 0,
}: FileTreeNodeProps) {
  const [expanded, setExpanded] = useState(true);
  const isDir = node.type === "directory";
  const isSelected = node.path === selectedPath;
  const paddingLeft = 12 + depth * 16;

  if (isDir) {
    return (
      <div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full text-left flex items-center gap-1.5 py-1 px-2 text-sm text-gray-400 hover:text-gray-200 hover:bg-gray-800/50 transition-colors"
          style={{ paddingLeft }}
        >
          <svg
            className={`w-3.5 h-3.5 flex-shrink-0 transition-transform ${expanded ? "rotate-90" : ""}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9 5l7 7-7 7"
            />
          </svg>
          <span className="truncate">{node.name}</span>
        </button>
        {expanded && node.children && (
          <div>
            {node.children.map((child) => (
              <FileTreeNode
                key={child.path}
                node={child}
                selectedPath={selectedPath}
                onSelect={onSelect}
                depth={depth + 1}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <button
      onClick={() => onSelect(node.path)}
      className={`w-full text-left flex items-center gap-1.5 py-1 px-2 text-sm transition-colors ${
        isSelected
          ? "bg-blue-900/30 text-blue-300"
          : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/50"
      }`}
      style={{ paddingLeft }}
    >
      <span className="text-[10px] font-mono text-gray-500 w-5 flex-shrink-0 text-center">
        {getFileIcon(node.name)}
      </span>
      <span className="truncate">{node.name}</span>
    </button>
  );
}
