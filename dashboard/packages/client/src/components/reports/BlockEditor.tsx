import { useEffect, useMemo } from "react";
import {
  useCreateBlockNote,
  SuggestionMenuController,
  getDefaultReactSlashMenuItems,
} from "@blocknote/react";
import { BlockNoteView } from "@blocknote/mantine";
import {
  BlockNoteSchema,
  defaultBlockSpecs,
  type Block,
} from "@blocknote/core";
import "@blocknote/mantine/style.css";

import { ExperimentChartBlock } from "./blocks/ExperimentChartBlock.js";
import { MetricsTableBlock } from "./blocks/MetricsTableBlock.js";
import { TrajectoryChartBlock } from "./blocks/TrajectoryChartBlock.js";
import { CrossPromptBlock } from "./blocks/CrossPromptBlock.js";
import { SweepBlock } from "./blocks/SweepBlock.js";

const schema = BlockNoteSchema.create({
  blockSpecs: {
    ...defaultBlockSpecs,
    experimentChart: ExperimentChartBlock(),
    metricsTable: MetricsTableBlock(),
    trajectoryChart: TrajectoryChartBlock(),
    crossPromptAnalysis: CrossPromptBlock(),
    sweepAnalysis: SweepBlock(),
  },
});

export type ReportBlock = Block<typeof schema.blockSchema>;

interface BlockEditorProps {
  initialBlocks?: ReportBlock[];
  onChange?: (blocks: ReportBlock[]) => void;
}

export default function BlockEditor({ initialBlocks, onChange }: BlockEditorProps) {
  const editor = useCreateBlockNote({
    schema,
    initialContent: initialBlocks && initialBlocks.length > 0
      ? initialBlocks
      : undefined,
  });

  // Sync external changes only once on mount if needed
  useEffect(() => {
    if (initialBlocks && initialBlocks.length > 0) {
      editor.replaceBlocks(editor.document, initialBlocks);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const getSlashMenuItems = useMemo(
    () => (query: string) => {
      const defaults = getDefaultReactSlashMenuItems(editor as any);

      const customItems = [
        {
          title: "Experiment Chart",
          subtext: "Insert a live chart from an experiment",
          group: "Experiment Data",
          onItemClick: () => {
            const current = editor.getTextCursorPosition().block;
            (editor as any).insertBlocks(
              [{ type: "experimentChart", props: { experimentId: "", metricKey: "Losses/Total", chartMode: "lm" } }],
              current,
              "after",
            );
          },
          aliases: ["chart", "plot", "graph"] as string[],
          key: "experiment_chart",
        },
        {
          title: "RL Training Chart",
          subtext: "Insert an RL training metrics chart",
          group: "Experiment Data",
          onItemClick: () => {
            const current = editor.getTextCursorPosition().block;
            (editor as any).insertBlocks(
              [{ type: "experimentChart", props: { experimentId: "", chartMode: "rl", rlMetricKey: "episode_rewards" } }],
              current,
              "after",
            );
          },
          aliases: ["rl", "reward", "ppo"] as string[],
          key: "rl_training_chart",
        },
        {
          title: "Ablation Chart",
          subtext: "Insert an ablation study comparison chart",
          group: "Experiment Data",
          onItemClick: () => {
            const current = editor.getTextCursorPosition().block;
            (editor as any).insertBlocks(
              [{ type: "experimentChart", props: { experimentId: "", chartMode: "ablation", ablationMetricKey: "mean_reward" } }],
              current,
              "after",
            );
          },
          aliases: ["ablation", "comparison", "baseline"] as string[],
          key: "ablation_chart",
        },
        {
          title: "Metrics Table",
          subtext: "Insert a summary metrics table from an experiment",
          group: "Experiment Data",
          onItemClick: () => {
            const current = editor.getTextCursorPosition().block;
            (editor as any).insertBlocks(
              [{ type: "metricsTable", props: { experimentId: "" } }],
              current,
              "after",
            );
          },
          aliases: ["metrics", "summary", "stats"] as string[],
          key: "metrics_table",
        },
        {
          title: "Gate Trajectory",
          subtext: "Insert a single-prompt gate signal trajectory chart",
          group: "Trajectory Analysis",
          onItemClick: () => {
            const current = editor.getTextCursorPosition().block;
            (editor as any).insertBlocks(
              [{ type: "trajectoryChart", props: { experimentId: "", gatingMode: "fixed", prompt: "Once upon a time," } }],
              current,
              "after",
            );
          },
          aliases: ["trajectory", "gate", "signals"] as string[],
          key: "trajectory_chart",
        },
        {
          title: "Cross-Prompt Analysis",
          subtext: "Compare gating strategies across multiple prompts",
          group: "Trajectory Analysis",
          onItemClick: () => {
            const current = editor.getTextCursorPosition().block;
            (editor as any).insertBlocks(
              [{ type: "crossPromptAnalysis", props: { experimentId: "", gatingMode: "fixed" } }],
              current,
              "after",
            );
          },
          aliases: ["cross-prompt", "heatmap", "comparison"] as string[],
          key: "cross_prompt_analysis",
        },
        {
          title: "Gate Sweep",
          subtext: "Run extreme-value sweep to map gate signal effects",
          group: "Trajectory Analysis",
          onItemClick: () => {
            const current = editor.getTextCursorPosition().block;
            (editor as any).insertBlocks(
              [{ type: "sweepAnalysis", props: { experimentId: "", prompt: "Once upon a time," } }],
              current,
              "after",
            );
          },
          aliases: ["sweep", "extreme", "interpretability"] as string[],
          key: "sweep_analysis",
        },
      ];

      const all = [...defaults, ...customItems];

      if (!query) return all;
      const q = query.toLowerCase();
      return all.filter(
        (item) =>
          item.title.toLowerCase().includes(q) ||
          (item.aliases ?? []).some((a: string) => a.toLowerCase().includes(q)) ||
          (item.subtext ?? "").toLowerCase().includes(q),
      );
    },
    [editor],
  );

  return (
    <BlockNoteView
      editor={editor}
      onChange={() => {
        onChange?.(editor.document as ReportBlock[]);
      }}
      slashMenu={false}
      theme="dark"
    >
      <SuggestionMenuController
        triggerCharacter="/"
        getItems={async (query) => getSlashMenuItems(query)}
      />
    </BlockNoteView>
  );
}
