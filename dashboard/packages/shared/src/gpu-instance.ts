/** Rich metadata from a Vast.ai GPU instance, captured at provision time. */
export interface VastInstanceMeta {
  instanceId: number;
  offerId: number;
  hostId: number | null;
  machineId: number | null;
  gpuName: string;
  numGpus: number;
  gpuRamMb: number | null;
  gpuMemBwGbps: number | null;
  totalFlops: number | null;
  dlPerf: number | null;
  dlPerfPerDphTotal: number | null;
  cpuName: string | null;
  cpuCores: number | null;
  cpuCoresEffective: number | null;
  cpuRamMb: number | null;
  diskName: string | null;
  diskBwMbps: number | null;
  diskSpaceGb: number | null;
  inetDownMbps: number | null;
  inetUpMbps: number | null;
  moboName: string | null;
  cudaMaxGood: number | null;
  reliability: number | null;
  costPerHour: number;
  capturedAt: number;
}

/** GET /api/experiments/:expId/gpu-instance */
export interface GpuInstanceResponse {
  expId: string;
  instance: VastInstanceMeta | null;
}
