import { useMemo } from "react";
import { useFullMetrics, useRLMetrics, useCheckpoints, useEvaluation } from "./useMetrics.js";
import { useConfigFiles, useConfigFile } from "./useConfigs.js";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client.js";
import type { ReportData } from "../utils/report.js";

export function useReportData(expId: string | null): ReportData | null {
  const { data: metricsData } = useFullMetrics(expId);
  const { data: rlData } = useRLMetrics(expId);
  const { data: checkpointsData } = useCheckpoints(expId);
  const { data: evalData } = useEvaluation(expId);
  const { data: configList } = useConfigFiles();
  const { data: baseConfig } = useConfigFile(configList?.files?.[0] ?? null);
  const { data: statusData } = useQuery({
    queryKey: ["status", expId],
    queryFn: async () => {
      const res = await api.getStatus(expId!);
      return res.status;
    },
    enabled: !!expId,
    staleTime: Infinity,
  });

  return useMemo(() => {
    if (!expId) return null;

    const points = metricsData?.points ?? [];
    const lastPoint = points.length > 0 ? points[points.length - 1] : null;

    return {
      experimentId: expId,
      configYAML: baseConfig?.content ?? "",
      metrics: {
        finalLoss: lastPoint ? (lastPoint["Losses/Total"] as number) ?? null : null,
        finalPerplexity: lastPoint
          ? Math.exp((lastPoint["Losses/Total"] as number) ?? 0)
          : null,
        totalSteps: lastPoint?.step ?? null,
        trainingStatus: statusData?.status ?? null,
      },
      checkpoints: (checkpointsData?.checkpoints ?? []).map((c) => ({
        filename: c.filename,
        phase: c.phase,
        epoch: c.epoch,
      })),
      samples: evalData?.results?.samples ?? [],
      rlMetrics: rlData?.metrics?.history
        ? {
            episodeCount: rlData.metrics.history.episode_rewards.length,
            finalReward:
              rlData.metrics.history.episode_rewards.length > 0
                ? rlData.metrics.history.episode_rewards[
                    rlData.metrics.history.episode_rewards.length - 1
                  ]
                : null,
          }
        : null,
    };
  }, [expId, metricsData, rlData, checkpointsData, evalData, baseConfig, statusData]);
}
