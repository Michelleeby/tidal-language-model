// ---------------------------------------------------------------------------
// Block Patterns — predefined compositions of BlockNote blocks
// ---------------------------------------------------------------------------

import type { BlockContent } from "./reports.js";

/** Metadata for a block pattern. */
export interface BlockPattern {
  name: string;
  description: string;
  build: (experimentId: string) => BlockContent[];
}

// ---------------------------------------------------------------------------
// Pattern builders
// ---------------------------------------------------------------------------

function buildExperimentOverview(experimentId: string): BlockContent[] {
  return [
    {
      id: crypto.randomUUID(),
      type: "heading",
      props: { level: 2 },
      content: [{ type: "text", text: "Experiment Overview" }],
      children: [],
    },
    {
      id: crypto.randomUUID(),
      type: "paragraph",
      content: [
        {
          type: "text",
          text: `Language model training overview for experiment ${experimentId}.`,
        },
      ],
      children: [],
    },
    {
      id: crypto.randomUUID(),
      type: "experimentChart",
      props: { experimentId, metricKey: "Losses/Total", chartMode: "lm" },
      children: [],
    },
    {
      id: crypto.randomUUID(),
      type: "experimentChart",
      props: { experimentId, metricKey: "Perplexity", chartMode: "lm" },
      children: [],
    },
    {
      id: crypto.randomUUID(),
      type: "metricsTable",
      props: { experimentId },
      children: [],
    },
  ];
}

function buildRlAnalysis(experimentId: string): BlockContent[] {
  return [
    {
      id: crypto.randomUUID(),
      type: "heading",
      props: { level: 2 },
      content: [{ type: "text", text: "RL Gating Analysis" }],
      children: [],
    },
    {
      id: crypto.randomUUID(),
      type: "paragraph",
      content: [
        {
          type: "text",
          text: `Reinforcement learning gating controller analysis for experiment ${experimentId}.`,
        },
      ],
      children: [],
    },
    {
      id: crypto.randomUUID(),
      type: "experimentChart",
      props: { experimentId, chartMode: "rl", rlMetricKey: "episode_rewards" },
      children: [],
    },
    {
      id: crypto.randomUUID(),
      type: "experimentChart",
      props: { experimentId, chartMode: "rl", rlMetricKey: "policy_loss" },
      children: [],
    },
    {
      id: crypto.randomUUID(),
      type: "experimentChart",
      props: { experimentId, chartMode: "rl", rlMetricKey: "gate_modulation" },
      children: [],
    },
    {
      id: crypto.randomUUID(),
      type: "experimentChart",
      props: { experimentId, chartMode: "ablation", ablationMetricKey: "mean_reward" },
      children: [],
    },
  ];
}

function buildTrajectoryReport(experimentId: string): BlockContent[] {
  return [
    {
      id: crypto.randomUUID(),
      type: "heading",
      props: { level: 2 },
      content: [{ type: "text", text: "Gate Trajectory Analysis" }],
      children: [],
    },
    {
      id: crypto.randomUUID(),
      type: "paragraph",
      content: [
        {
          type: "text",
          text: `Gate signal trajectory analysis for experiment ${experimentId}.`,
        },
      ],
      children: [],
    },
    {
      id: crypto.randomUUID(),
      type: "trajectoryChart",
      props: { experimentId, gatingMode: "fixed", prompt: "Once upon a time," },
      children: [],
    },
    {
      id: crypto.randomUUID(),
      type: "crossPromptAnalysis",
      props: { experimentId, gatingMode: "fixed" },
      children: [],
    },
    {
      id: crypto.randomUUID(),
      type: "sweepAnalysis",
      props: { experimentId, prompt: "Once upon a time," },
      children: [],
    },
  ];
}

function buildFullReport(experimentId: string): BlockContent[] {
  return [
    {
      id: crypto.randomUUID(),
      type: "heading",
      props: { level: 1 },
      content: [{ type: "text", text: `Full Report — ${experimentId}` }],
      children: [],
    },
    ...buildExperimentOverview(experimentId),
    ...buildRlAnalysis(experimentId),
    ...buildTrajectoryReport(experimentId),
  ];
}

// ---------------------------------------------------------------------------
// Pattern registry
// ---------------------------------------------------------------------------

export const BLOCK_PATTERNS: readonly BlockPattern[] = [
  {
    name: "experiment-overview",
    description: "LM training overview with loss chart, perplexity chart, and metrics summary table",
    build: buildExperimentOverview,
  },
  {
    name: "rl-analysis",
    description: "RL gating analysis with episode rewards, policy loss, gate signal charts, and ablation comparison",
    build: buildRlAnalysis,
  },
  {
    name: "trajectory-report",
    description: "Gate trajectory analysis with single-prompt trajectory, cross-prompt heatmap, and extreme-value sweep",
    build: buildTrajectoryReport,
  },
  {
    name: "full-report",
    description: "Complete report combining experiment overview, RL analysis, and trajectory analysis",
    build: buildFullReport,
  },
] as const;

/** Build blocks for a named pattern. Returns null if pattern not found. */
export function buildPatternBlocks(
  patternName: string,
  experimentId: string,
): BlockContent[] | null {
  const pattern = BLOCK_PATTERNS.find((p) => p.name === patternName);
  return pattern ? pattern.build(experimentId) : null;
}

/** List all available pattern names with descriptions. */
export function listPatterns(): Array<{ name: string; description: string }> {
  return BLOCK_PATTERNS.map((p) => ({ name: p.name, description: p.description }));
}
