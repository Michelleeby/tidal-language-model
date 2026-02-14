import { useState } from "react";
import GenerationPanel from "./GenerationPanel.js";

interface GenerationComparisonProps {
  expId: string;
}

export default function GenerationComparison({
  expId,
}: GenerationComparisonProps) {
  const [prompt, setPrompt] = useState("Once upon a time");

  return (
    <div className="space-y-4">
      {/* Shared prompt */}
      <div>
        <label className="block text-xs text-gray-500 mb-1">
          Shared Prompt
        </label>
        <textarea
          className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200 resize-none"
          rows={3}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
      </div>

      {/* Side-by-side panels */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="border border-gray-800 rounded-lg p-4">
          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Panel A
          </h4>
          <GenerationPanel
            expId={expId}
            prompt={prompt}
            onPromptChange={setPrompt}
            sharedPrompt
          />
        </div>
        <div className="border border-gray-800 rounded-lg p-4">
          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Panel B
          </h4>
          <GenerationPanel
            expId={expId}
            prompt={prompt}
            onPromptChange={setPrompt}
            sharedPrompt
          />
        </div>
      </div>
    </div>
  );
}
