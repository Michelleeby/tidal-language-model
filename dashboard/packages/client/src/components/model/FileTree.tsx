import type { PluginFileNode } from "@tidal/shared";
import FileTreeNode from "./FileTreeNode.js";

interface FileTreeProps {
  files: PluginFileNode[];
  selectedPath: string | null;
  onSelect: (path: string) => void;
}

export default function FileTree({ files, selectedPath, onSelect }: FileTreeProps) {
  if (files.length === 0) {
    return (
      <div className="px-4 py-3 text-sm text-gray-500">No files</div>
    );
  }

  return (
    <div className="py-1">
      {files.map((node) => (
        <FileTreeNode
          key={node.path}
          node={node}
          selectedPath={selectedPath}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}
