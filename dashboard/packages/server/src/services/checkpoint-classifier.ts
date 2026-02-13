import type { CheckpointPattern } from "@tidal/shared";

export interface ClassifiedCheckpoint {
  phase: string;
  epoch?: number;
}

/**
 * Convert a manifest glob pattern to a RegExp.
 * Only supports `*` (match any non-path chars).
 */
function globToRegex(glob: string): RegExp {
  const escaped = glob.replace(/[.+^${}()|[\]\\]/g, "\\$&");
  const pattern = escaped.replace(/\*/g, ".*");
  return new RegExp(`^${pattern}$`);
}

/**
 * Classify a checkpoint filename using the plugin's checkpoint patterns.
 * Returns the phase and optional epoch/iteration number.
 */
export function classifyCheckpoint(
  filename: string,
  patterns: CheckpointPattern[],
): ClassifiedCheckpoint {
  for (const pattern of patterns) {
    // Check excludePrefix first
    if (pattern.excludePrefix && filename.startsWith(pattern.excludePrefix)) {
      continue;
    }

    const regex = globToRegex(pattern.glob);
    if (!regex.test(filename)) continue;

    let epoch: number | undefined;
    if (pattern.epochCapture) {
      const match = filename.match(new RegExp(pattern.epochCapture));
      if (match?.[1]) {
        epoch = parseInt(match[1], 10);
      }
    }

    return { phase: pattern.phase, epoch };
  }

  return { phase: "unknown" };
}
