import { useState } from "react";

interface ChartExportData {
  headers: string[];
  rows: number[][];
}

interface ChartExportButtonProps {
  data: ChartExportData;
  filename: string;
}

function toCSV(data: ChartExportData): string {
  const lines = [data.headers.join(",")];
  for (const row of data.rows) {
    lines.push(row.join(","));
  }
  return lines.join("\n");
}

function toJSON(data: ChartExportData): string {
  const records = data.rows.map((row) => {
    const obj: Record<string, number> = {};
    data.headers.forEach((h, i) => {
      obj[h] = row[i];
    });
    return obj;
  });
  return JSON.stringify(records, null, 2);
}

function download(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ChartExportButton({
  data,
  filename,
}: ChartExportButtonProps) {
  const [open, setOpen] = useState(false);

  if (data.rows.length === 0) return null;

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="p-1 text-gray-500 hover:text-gray-300 transition-colors"
        title="Export data"
      >
        <svg
          className="w-4 h-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
          />
        </svg>
      </button>
      {open && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setOpen(false)}
          />
          <div className="absolute right-0 mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-lg z-20 py-1 min-w-[130px]">
            <button
              onClick={() => {
                download(toCSV(data), `${filename}.csv`, "text/csv");
                setOpen(false);
              }}
              className="w-full text-left px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-700"
            >
              Download CSV
            </button>
            <button
              onClick={() => {
                download(
                  toJSON(data),
                  `${filename}.json`,
                  "application/json",
                );
                setOpen(false);
              }}
              className="w-full text-left px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-700"
            >
              Download JSON
            </button>
          </div>
        </>
      )}
    </div>
  );
}
