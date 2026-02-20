import { useState } from "react";
import { useExperiments } from "../../hooks/useExperiments.js";
import { useCreateJob } from "../../hooks/useJobs.js";
import { useAllLMCheckpoints } from "../../hooks/useMetrics.js";
import { usePlugin } from "../../hooks/usePlugin.js";
import { hasCompletedLMExperiment } from "../../utils/experimentFilters.js";
import { filterRLEligibleCheckpoints } from "./RLTrainingTrigger.js";

type Tab = "lm" | "rl";

export default function NewExperimentPanel() {
  const { data: expData } = useExperiments();
  const { data: allCheckpointsData } = useAllLMCheckpoints();
  const createJob = useCreateJob();
  const { manifest } = usePlugin();

  const experiments = expData?.experiments ?? [];
  const canStartRL = hasCompletedLMExperiment(experiments);
  const [activeTab, setActiveTab] = useState<Tab>("lm");

  // LM config defaults from manifest
  const lmPhase = manifest?.trainingPhases.find((p) => p.id === "lm-training");
  const pluginPrefix = manifest ? `plugins/${manifest.name}/` : "plugins/tidal/";
  const defaultLMConfigPath = lmPhase?.configFiles[0]
    ? `${pluginPrefix}${lmPhase.configFiles[0]}`
    : `${pluginPrefix}configs/base_config.yaml`;

  // RL config defaults from manifest
  const rlPhase = manifest?.trainingPhases.find((p) => p.id === "rl-training");
  const defaultRLBaseConfigPath = rlPhase?.configFiles[0]
    ? `${pluginPrefix}${rlPhase.configFiles[0]}`
    : `${pluginPrefix}configs/base_config.yaml`;
  const defaultRLConfigPath = rlPhase?.configFiles[1]
    ? `${pluginPrefix}${rlPhase.configFiles[1]}`
    : `${pluginPrefix}configs/rl_config.yaml`;

  // LM form state
  const [lmConfigPath, setLmConfigPath] = useState(defaultLMConfigPath);
  const [resumeDir, setResumeDir] = useState("");

  // RL form state
  const [rlBaseConfigPath, setRlBaseConfigPath] = useState(defaultRLBaseConfigPath);
  const [rlConfigPath, setRlConfigPath] = useState(defaultRLConfigPath);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
  const [timesteps, setTimesteps] = useState("");

  // RL-eligible checkpoints grouped by experiment
  const groups = allCheckpointsData?.groups ?? [];
  const eligibleGroups = groups
    .map((g) => ({
      ...g,
      checkpoints: filterRLEligibleCheckpoints(g.checkpoints),
    }))
    .filter((g) => g.checkpoints.length > 0);

  const handleStartLM = () => {
    createJob.mutate({
      type: "lm-training",
      plugin: manifest?.name ?? "tidal",
      configPath: lmConfigPath,
      resumeExpDir: resumeDir || undefined,
    });
  };

  const handleStartRL = () => {
    if (!selectedCheckpoint) return;
    createJob.mutate({
      type: "rl-training",
      plugin: manifest?.name ?? "tidal",
      configPath: rlBaseConfigPath,
      rlConfigPath,
      checkpoint: selectedCheckpoint,
      timesteps: timesteps ? parseInt(timesteps, 10) : undefined,
    });
  };

  return (
    <div className="space-y-4">
      {/* Tab toggle */}
      <div className="flex gap-1 bg-gray-800 rounded-lg p-1 w-fit">
        <button
          onClick={() => setActiveTab("lm")}
          className={`px-4 py-1.5 text-sm rounded-md transition-colors ${
            activeTab === "lm"
              ? "bg-blue-600 text-white"
              : "text-gray-400 hover:text-gray-200"
          }`}
        >
          Language Model
        </button>
        <button
          onClick={() => canStartRL && setActiveTab("rl")}
          disabled={!canStartRL}
          className={`px-4 py-1.5 text-sm rounded-md transition-colors ${
            activeTab === "rl"
              ? "bg-purple-600 text-white"
              : canStartRL
                ? "text-gray-400 hover:text-gray-200"
                : "text-gray-600 cursor-not-allowed"
          }`}
        >
          RL Gating
        </button>
      </div>

      {!canStartRL && activeTab === "lm" && (
        <p className="text-xs text-gray-600">
          Complete an LM training run to unlock RL gating.
        </p>
      )}

      {/* LM tab */}
      {activeTab === "lm" && (
        <div className="space-y-3">
          <div className="flex flex-col gap-2">
            <label className="text-xs text-gray-500">Config path</label>
            <input
              type="text"
              value={lmConfigPath}
              onChange={(e) => setLmConfigPath(e.target.value)}
              placeholder="Config path"
              className="bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-gray-200 w-full focus:outline-none focus:border-blue-500"
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className="text-xs text-gray-500">Resume directory (optional)</label>
            <input
              type="text"
              value={resumeDir}
              onChange={(e) => setResumeDir(e.target.value)}
              placeholder="experiments/..."
              className="bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-gray-200 w-full focus:outline-none focus:border-blue-500"
            />
          </div>
          <button
            onClick={handleStartLM}
            disabled={createJob.isPending || !lmConfigPath}
            className="px-4 py-2 text-sm rounded bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors w-full"
          >
            {createJob.isPending ? "Starting..." : "Start LM Training"}
          </button>
        </div>
      )}

      {/* RL tab */}
      {activeTab === "rl" && (
        <div className="space-y-3">
          <div className="flex flex-col gap-2">
            <label className="text-xs text-gray-500">Model checkpoint</label>
            <select
              value={selectedCheckpoint}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
              className="bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-gray-200 w-full focus:outline-none focus:border-blue-500"
            >
              <option value="">Select model checkpoint</option>
              {eligibleGroups.map((group) => (
                <optgroup key={group.experimentId} label={group.experimentId}>
                  {group.checkpoints.map((cp) => (
                    <option key={cp.path} value={cp.path}>
                      {cp.filename}
                      {cp.epoch != null ? ` (epoch ${cp.epoch})` : ""}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>
          <div className="flex flex-col gap-2">
            <label className="text-xs text-gray-500">Base config</label>
            <input
              type="text"
              value={rlBaseConfigPath}
              onChange={(e) => setRlBaseConfigPath(e.target.value)}
              className="bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-gray-200 w-full focus:outline-none focus:border-blue-500"
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className="text-xs text-gray-500">RL config</label>
            <input
              type="text"
              value={rlConfigPath}
              onChange={(e) => setRlConfigPath(e.target.value)}
              className="bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-gray-200 w-full focus:outline-none focus:border-blue-500"
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className="text-xs text-gray-500">Timesteps (optional)</label>
            <input
              type="text"
              value={timesteps}
              onChange={(e) => setTimesteps(e.target.value.replace(/\D/g, ""))}
              placeholder="e.g. 50000"
              className="bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-gray-200 w-full focus:outline-none focus:border-blue-500"
            />
          </div>
          <button
            onClick={handleStartRL}
            disabled={!selectedCheckpoint || createJob.isPending}
            className="px-4 py-2 text-sm rounded bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors w-full"
          >
            {createJob.isPending ? "Starting..." : "Start RL Training"}
          </button>
        </div>
      )}

      {createJob.isError && (
        <p className="text-sm text-red-400">
          {(createJob.error as Error).message}
        </p>
      )}
    </div>
  );
}
