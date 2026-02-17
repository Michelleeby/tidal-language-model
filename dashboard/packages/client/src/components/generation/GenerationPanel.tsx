import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useCheckpoints } from "../../hooks/useMetrics.js";
import { usePlugin } from "../../hooks/usePlugin.js";
import { api } from "../../api/client.js";
import type { GenerateRequest, GenerationMode } from "@tidal/shared";
import GateTrajectoryChart from "../charts/GateTrajectoryChart.js";

const DEFAULT_MODES: GenerationMode[] = [
  { id: "none", displayName: "None", requiresRLCheckpoint: false },
  { id: "random", displayName: "Random", requiresRLCheckpoint: false },
  { id: "fixed", displayName: "Fixed", requiresRLCheckpoint: false },
  { id: "learned", displayName: "Learned (RL)", requiresRLCheckpoint: true },
];

interface GenerationPanelProps {
  expId: string;
  prompt: string;
  onPromptChange?: (prompt: string) => void;
  sharedPrompt?: boolean;
}

export default function GenerationPanel({
  expId,
  prompt: externalPrompt,
  onPromptChange,
  sharedPrompt = false,
}: GenerationPanelProps) {
  const { manifest } = usePlugin();
  const { data: checkpointsData } = useCheckpoints(expId);

  const modes = manifest?.generation.modes ?? DEFAULT_MODES;
  const params = manifest?.generation.parameters ?? [];

  const [localPrompt, setLocalPrompt] = useState("Once upon a time");
  const prompt = sharedPrompt ? externalPrompt : localPrompt;
  const setPrompt = sharedPrompt
    ? (v: string) => onPromptChange?.(v)
    : setLocalPrompt;

  const [checkpoint, setCheckpoint] = useState("");
  const [gatingMode, setGatingMode] = useState<string>("none");

  const [paramValues, setParamValues] = useState<Record<string, number>>(
    () => {
      const defaults: Record<string, number> = {};
      for (const p of params) defaults[p.id] = p.default;
      if (!defaults.temperature) defaults.temperature = 0.8;
      if (!defaults.topK) defaults.topK = 50;
      if (!defaults.maxTokens) defaults.maxTokens = 50;
      return defaults;
    },
  );

  const setParam = (id: string, value: number) => {
    setParamValues((prev) => ({ ...prev, [id]: value }));
  };

  const generateMutation = useMutation({
    mutationFn: (body: GenerateRequest) => api.generate(body),
  });

  const handleGenerate = () => {
    if (!checkpoint) return;
    generateMutation.mutate({
      checkpoint,
      prompt,
      maxTokens: paramValues.maxTokens ?? 50,
      temperature: paramValues.temperature ?? 0.8,
      topK: paramValues.topK ?? 50,
      gatingMode: gatingMode as GenerateRequest["gatingMode"],
    });
  };

  const learnedMode = modes.find((m) => m.requiresRLCheckpoint);

  return (
    <div className="space-y-4">
      {/* Checkpoint selection */}
      <div>
        <label className="block text-xs text-gray-500 mb-1">Checkpoint</label>
        <select
          className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200"
          value={checkpoint}
          onChange={(e) => {
            setCheckpoint(e.target.value);
            const selected = checkpointsData?.checkpoints.find(
              (cp) => cp.path === e.target.value,
            );
            if (selected?.phase === "rl" && learnedMode) {
              setGatingMode(learnedMode.id);
            }
          }}
        >
          <option value="">Select checkpoint</option>
          {checkpointsData?.checkpoints.map((cp) => (
            <option key={cp.filename} value={cp.path}>
              {cp.filename} ({cp.phase})
            </option>
          ))}
        </select>
      </div>

      {/* Prompt — only show if not shared */}
      {!sharedPrompt && (
        <div>
          <label className="block text-xs text-gray-500 mb-1">Prompt</label>
          <textarea
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200 resize-none"
            rows={3}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>
      )}

      {/* Parameters */}
      <div className="grid grid-cols-2 gap-3">
        {params.length > 0
          ? params.map((p) => (
              <div key={p.id}>
                <label className="block text-xs text-gray-500 mb-1">
                  {p.displayName}: {paramValues[p.id] ?? p.default}
                </label>
                <input
                  type="range"
                  min={p.min}
                  max={p.max}
                  step={p.step}
                  value={paramValues[p.id] ?? p.default}
                  onChange={(e) => setParam(p.id, parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            ))
          : (
              <>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">
                    Temperature: {paramValues.temperature}
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="2.0"
                    step="0.1"
                    value={paramValues.temperature}
                    onChange={(e) =>
                      setParam("temperature", parseFloat(e.target.value))
                    }
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">
                    Top-K: {paramValues.topK}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="200"
                    step="1"
                    value={paramValues.topK}
                    onChange={(e) =>
                      setParam("topK", parseInt(e.target.value))
                    }
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">
                    Max Tokens: {paramValues.maxTokens}
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="200"
                    step="10"
                    value={paramValues.maxTokens}
                    onChange={(e) =>
                      setParam("maxTokens", parseInt(e.target.value))
                    }
                    className="w-full"
                  />
                </div>
              </>
            )}

        <div>
          <label className="block text-xs text-gray-500 mb-1">
            Gating Mode
          </label>
          <select
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200"
            value={gatingMode}
            onChange={(e) => setGatingMode(e.target.value)}
          >
            {modes.map((m) => (
              <option key={m.id} value={m.id}>
                {m.displayName}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Generate button */}
      <button
        onClick={handleGenerate}
        disabled={!checkpoint || generateMutation.isPending}
        className="px-4 py-2 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {generateMutation.isPending ? "Generating..." : "Generate"}
      </button>

      {/* Output */}
      {generateMutation.data && (
        <div className="bg-gray-800 rounded-lg p-3 space-y-2">
          <div className="flex justify-between text-xs text-gray-500">
            <span>{generateMutation.data.tokensGenerated} tokens</span>
            <span>{generateMutation.data.elapsedMs}ms</span>
          </div>
          <div className="font-mono text-sm text-gray-100 whitespace-pre-wrap">
            {generateMutation.data.text}
          </div>
        </div>
      )}

      {generateMutation.data?.trajectory && (
        <GateTrajectoryChart trajectory={generateMutation.data.trajectory} />
      )}

      {generateMutation.isError && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg p-3 text-sm text-red-300">
          {(generateMutation.error as Error).message}
        </div>
      )}
    </div>
  );
}
