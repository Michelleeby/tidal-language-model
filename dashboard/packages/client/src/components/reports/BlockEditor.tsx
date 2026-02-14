import { useEffect } from "react";
import { useCreateBlockNote } from "@blocknote/react";
import { BlockNoteView } from "@blocknote/mantine";
import {
  BlockNoteSchema,
  defaultBlockSpecs,
  type Block,
} from "@blocknote/core";
import "@blocknote/mantine/style.css";

import { ExperimentChartBlock } from "./blocks/ExperimentChartBlock.js";
import { MetricsTableBlock } from "./blocks/MetricsTableBlock.js";

const schema = BlockNoteSchema.create({
  blockSpecs: {
    ...defaultBlockSpecs,
    experimentChart: ExperimentChartBlock(),
    metricsTable: MetricsTableBlock(),
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

  return (
    <BlockNoteView
      editor={editor}
      onChange={() => {
        onChange?.(editor.document as ReportBlock[]);
      }}
      theme="dark"
    />
  );
}
