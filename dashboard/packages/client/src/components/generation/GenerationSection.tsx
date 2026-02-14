import { useState } from "react";
import GenerationPanel from "./GenerationPanel.js";
import GenerationComparison from "./GenerationComparison.js";

interface GenerationSectionProps {
  expId: string;
}

export default function GenerationSection({ expId }: GenerationSectionProps) {
  const [mode, setMode] = useState<"single" | "compare">("single");

  return (
    <div className="space-y-4">
      {/* Mode toggle */}
      <div className="flex gap-1">
        <button
          onClick={() => setMode("single")}
          className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${
            mode === "single"
              ? "bg-gray-700 text-gray-100"
              : "text-gray-500 hover:text-gray-300 hover:bg-gray-800"
          }`}
        >
          Single
        </button>
        <button
          onClick={() => setMode("compare")}
          className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${
            mode === "compare"
              ? "bg-gray-700 text-gray-100"
              : "text-gray-500 hover:text-gray-300 hover:bg-gray-800"
          }`}
        >
          Compare
        </button>
      </div>

      {mode === "single" ? (
        <GenerationPanel expId={expId} prompt="Once upon a time" />
      ) : (
        <GenerationComparison expId={expId} />
      )}
    </div>
  );
}
