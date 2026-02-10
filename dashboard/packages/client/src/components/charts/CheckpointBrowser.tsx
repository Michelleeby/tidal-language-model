import type { CheckpointInfo } from "@tidal/shared";

interface CheckpointBrowserProps {
  checkpoints: CheckpointInfo[];
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function CheckpointBrowser({
  checkpoints,
}: CheckpointBrowserProps) {
  if (checkpoints.length === 0) {
    return (
      <div className="text-gray-500 text-sm p-4">No checkpoints found</div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-gray-500 border-b border-gray-800">
            <th className="px-4 py-2">Filename</th>
            <th className="px-4 py-2">Phase</th>
            <th className="px-4 py-2">Epoch/Iter</th>
            <th className="px-4 py-2">Size</th>
            <th className="px-4 py-2">Modified</th>
          </tr>
        </thead>
        <tbody>
          {checkpoints.map((cp) => (
            <tr
              key={cp.filename}
              className="border-b border-gray-800/50 hover:bg-gray-800/30"
            >
              <td className="px-4 py-2 font-mono text-gray-300">
                {cp.filename}
              </td>
              <td className="px-4 py-2">
                <span
                  className={`text-xs px-2 py-0.5 rounded ${
                    cp.phase === "foundational"
                      ? "bg-blue-900/50 text-blue-300"
                      : cp.phase === "rl"
                        ? "bg-green-900/50 text-green-300"
                        : "bg-purple-900/50 text-purple-300"
                  }`}
                >
                  {cp.phase}
                </span>
              </td>
              <td className="px-4 py-2 text-gray-400">
                {cp.epoch ?? "—"}
              </td>
              <td className="px-4 py-2 text-gray-400 font-mono">
                {formatBytes(cp.sizeBytes)}
              </td>
              <td className="px-4 py-2 text-gray-500 text-xs">
                {new Date(cp.modified).toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
