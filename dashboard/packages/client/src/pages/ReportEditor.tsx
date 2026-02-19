import { useState, useEffect, useRef, useCallback } from "react";
import { useExperimentStore } from "../stores/experimentStore.js";
import { useReport, useUpdateReport } from "../hooks/useReports.js";
import BlockEditor from "../components/reports/BlockEditor.js";
import type { ReportBlock } from "../components/reports/BlockEditor.js";
import {
  exportToHTML,
  exportToMarkdown,
  exportToInteractiveHTML,
  captureBlockCanvases,
} from "../utils/reportExport.js";
import { api } from "../api/client.js";

type SaveStatus = "saved" | "saving" | "unsaved";

export default function ReportEditor() {
  const { selectedReportId } = useExperimentStore();
  const { data, isLoading } = useReport(selectedReportId);
  const updateReport = useUpdateReport();

  const [title, setTitle] = useState("");
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("saved");
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const blocksRef = useRef<ReportBlock[]>([]);
  const titleRef = useRef(title);
  titleRef.current = title;

  // Sync title from server data
  useEffect(() => {
    if (data?.report) {
      setTitle(data.report.title);
      blocksRef.current = (data.report.blocks ?? []) as ReportBlock[];
      setSaveStatus("saved");
    }
  }, [data]);

  const save = useCallback(
    async (overrides?: { title?: string; blocks?: ReportBlock[] }) => {
      if (!selectedReportId) return;
      setSaveStatus("saving");
      try {
        await updateReport.mutateAsync({
          id: selectedReportId,
          title: overrides?.title ?? titleRef.current,
          blocks: overrides?.blocks ?? blocksRef.current,
        });
        setSaveStatus("saved");
      } catch {
        setSaveStatus("unsaved");
      }
    },
    [selectedReportId, updateReport],
  );

  const debouncedSave = useCallback(
    (overrides?: { title?: string; blocks?: ReportBlock[] }) => {
      setSaveStatus("unsaved");
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => save(overrides), 1500);
    },
    [save],
  );

  const handleTitleChange = (newTitle: string) => {
    setTitle(newTitle);
    debouncedSave({ title: newTitle });
  };

  const handleBlocksChange = (blocks: ReportBlock[]) => {
    blocksRef.current = blocks;
    debouncedSave({ blocks });
  };

  const editorContainerRef = useRef<HTMLDivElement>(null);

  /** Collect analysis data for all blocks that have an analysisId. */
  const collectAnalysisData = async (): Promise<Map<string, Record<string, unknown>>> => {
    const dataMap = new Map<string, Record<string, unknown>>();
    const blocks = blocksRef.current;
    const fetches = blocks
      .filter((b) => (b.props as any)?.analysisId)
      .map(async (b) => {
        try {
          const resp = await api.getAnalysis((b.props as any).analysisId);
          dataMap.set(b.id, resp.analysis.data);
        } catch {
          // Skip blocks whose analysis cannot be fetched
        }
      });
    await Promise.all(fetches);
    return dataMap;
  };

  const handleExportHTML = async () => {
    setExportMenuOpen(false);
    const analysisData = await collectAnalysisData();
    // Use interactive export if we have any analysis data, otherwise basic
    const html = analysisData.size > 0
      ? exportToInteractiveHTML(titleRef.current, blocksRef.current, analysisData)
      : exportToHTML(titleRef.current, blocksRef.current);
    downloadFile(`${titleRef.current || "report"}.html`, html, "text/html");
  };

  const handleExportMarkdown = async () => {
    setExportMenuOpen(false);
    // Capture canvases from the live editor DOM
    const captures = editorContainerRef.current
      ? captureBlockCanvases(editorContainerRef.current)
      : undefined;
    const md = exportToMarkdown(titleRef.current, blocksRef.current, captures);
    downloadFile(`${titleRef.current || "report"}.md`, md, "text/markdown");
  };

  if (!selectedReportId) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Select a report or create a new one
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Loading report...
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header bar */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <input
            type="text"
            value={title}
            onChange={(e) => handleTitleChange(e.target.value)}
            placeholder="Untitled Report"
            className="bg-transparent text-xl font-semibold text-gray-100 border-none outline-none flex-1 min-w-0 placeholder-gray-600"
          />
          <span
            className={`text-xs px-2 py-0.5 rounded flex-shrink-0 ${
              saveStatus === "saved"
                ? "text-green-400 bg-green-900/30"
                : saveStatus === "saving"
                  ? "text-yellow-400 bg-yellow-900/30"
                  : "text-gray-400 bg-gray-800"
            }`}
          >
            {saveStatus === "saved"
              ? "Saved"
              : saveStatus === "saving"
                ? "Saving..."
                : "Unsaved"}
          </span>
        </div>

        {/* Export dropdown */}
        <div className="relative ml-3">
          <button
            onClick={() => setExportMenuOpen(!exportMenuOpen)}
            className="text-sm px-3 py-1.5 rounded bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-700"
          >
            Export
          </button>
          {exportMenuOpen && (
            <>
              <div
                className="fixed inset-0 z-10"
                onClick={() => setExportMenuOpen(false)}
              />
              <div className="absolute right-0 top-full mt-1 z-20 bg-gray-800 border border-gray-700 rounded shadow-lg py-1 min-w-[140px]">
                <button
                  onClick={handleExportHTML}
                  className="w-full text-left px-3 py-1.5 text-sm text-gray-300 hover:bg-gray-700"
                >
                  HTML
                </button>
                <button
                  onClick={handleExportMarkdown}
                  className="w-full text-left px-3 py-1.5 text-sm text-gray-300 hover:bg-gray-700"
                >
                  Markdown
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Block editor */}
      <div ref={editorContainerRef}>
        <BlockEditor
          initialBlocks={(data?.report?.blocks ?? []) as ReportBlock[]}
          onChange={handleBlocksChange}
        />
      </div>
    </div>
  );
}

function downloadFile(filename: string, content: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
