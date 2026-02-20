import type { ExperimentSummary } from "@tidal/shared";

/**
 * Returns true if at least one experiment is a completed LM experiment
 * with available checkpoints — the prerequisite for starting RL training.
 */
export function hasCompletedLMExperiment(
  experiments: ExperimentSummary[],
): boolean {
  return experiments.some(
    (e) =>
      e.experimentType === "lm" &&
      e.status?.status === "completed" &&
      e.checkpoints.length > 0,
  );
}
