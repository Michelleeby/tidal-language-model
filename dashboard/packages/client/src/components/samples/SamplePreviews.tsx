import type { EvaluationResults } from "@tidal/shared";

interface SamplePreviewsProps {
  results: EvaluationResults | null;
}

export default function SamplePreviews({ results }: SamplePreviewsProps) {
  if (!results || !results.samples || results.samples.length === 0) {
    return (
      <div className="text-gray-500 text-sm p-4">
        No evaluation samples available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {results.perplexity !== undefined && (
        <div className="bg-gray-900 rounded-lg p-4">
          <span className="text-xs text-gray-500">Test Perplexity:</span>{" "}
          <span className="text-lg font-mono text-purple-400">
            {results.perplexity.toFixed(2)}
          </span>
        </div>
      )}
      {results.samples.map((sample, i) => (
        <div key={i} className="bg-gray-900 rounded-lg p-4 space-y-2">
          <div className="text-xs text-gray-500">
            Prompt{sample.temperature ? ` (T=${sample.temperature})` : ""}
          </div>
          <div className="text-sm text-gray-300 font-mono">{sample.prompt}</div>
          <div className="border-t border-gray-800 pt-2">
            <div className="text-xs text-gray-500">Generated</div>
            <div className="text-sm text-gray-100 font-mono whitespace-pre-wrap">
              {sample.generated}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
