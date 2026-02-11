import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useExperiments } from "../hooks/useExperiments.js";
import { useCheckpoints } from "../hooks/useMetrics.js";
import { api } from "../api/client.js";
import type { GenerateRequest } from "@tidal/shared";

export default function PlaygroundPage() {
  const { data: expData } = useExperiments();
  const [expId, setExpId] = useState<string | null>(null);
  const { data: checkpointsData } = useCheckpoints(expId);

  const [prompt, setPrompt] = useState("Once upon a time");
  const [checkpoint, setCheckpoint] = useState("");
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);
  const [maxTokens, setMaxTokens] = useState(50);
  const [gatingMode, setGatingMode] = useState<GenerateRequest["gatingMode"]>("none");

  const generateMutation = useMutation({
    mutationFn: (body: GenerateRequest) => api.generate(body),
  });

  const handleGenerate = () => {
    if (!checkpoint) return;
    generateMutation.mutate({
      checkpoint,
      prompt,
      maxTokens,
      temperature,
      topK,
      gatingMode,
    });
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <h2 className="text-lg font-semibold">Generation Playground</h2>

      {/* Experiment + checkpoint selection */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-gray-500 mb-1">
            Experiment
          </label>
          <select
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200"
            value={expId ?? ""}
            onChange={(e) => {
              setExpId(e.target.value || null);
              setCheckpoint("");
            }}
          >
            <option value="">Select experiment</option>
            {expData?.experiments.map((exp) => (
              <option key={exp.id} value={exp.id}>
                {exp.id}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">
            Checkpoint
          </label>
          <select
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200"
            value={checkpoint}
            onChange={(e) => {
              setCheckpoint(e.target.value);
              const filename = e.target.value.split("/").pop() ?? "";
              if (filename.startsWith("rl_checkpoint")) {
                setGatingMode("learned");
              }
            }}
            disabled={!expId}
          >
            <option value="">Select checkpoint</option>
            {checkpointsData?.checkpoints.map((cp) => (
              <option key={cp.filename} value={cp.path}>
                {cp.filename} ({cp.phase})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Prompt */}
      <div>
        <label className="block text-xs text-gray-500 mb-1">Prompt</label>
        <textarea
          className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200 resize-none"
          rows={3}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
      </div>

      {/* Parameters */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <label className="block text-xs text-gray-500 mb-1">
            Temperature: {temperature}
          </label>
          <input
            type="range"
            min="0.1"
            max="2.0"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">
            Top-K: {topK}
          </label>
          <input
            type="range"
            min="1"
            max="200"
            step="1"
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">
            Max Tokens: {maxTokens}
          </label>
          <input
            type="range"
            min="10"
            max="200"
            step="10"
            value={maxTokens}
            onChange={(e) => setMaxTokens(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">
            Gating Mode
          </label>
          <select
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200"
            value={gatingMode}
            onChange={(e) =>
              setGatingMode(e.target.value as GenerateRequest["gatingMode"])
            }
          >
            <option value="none">None</option>
            <option value="random">Random</option>
            <option value="fixed">Fixed</option>
            <option value="learned">Learned (RL)</option>
          </select>
        </div>
      </div>

      {/* Generate button */}
      <button
        onClick={handleGenerate}
        disabled={!checkpoint || generateMutation.isPending}
        className="px-6 py-2 bg-blue-600 text-white rounded font-medium text-sm hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {generateMutation.isPending ? "Generating..." : "Generate"}
      </button>

      {/* Output */}
      {generateMutation.data && (
        <div className="bg-gray-900 rounded-lg p-4 space-y-2">
          <div className="flex justify-between text-xs text-gray-500">
            <span>{generateMutation.data.tokensGenerated} tokens</span>
            <span>{generateMutation.data.elapsedMs}ms</span>
          </div>
          <div className="font-mono text-sm text-gray-100 whitespace-pre-wrap">
            {generateMutation.data.text}
          </div>
        </div>
      )}

      {generateMutation.isError && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg p-4 text-sm text-red-300">
          {(generateMutation.error as Error).message}
        </div>
      )}
    </div>
  );
}
