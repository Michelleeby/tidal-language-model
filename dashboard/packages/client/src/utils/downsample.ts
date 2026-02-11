import type { MetricPoint } from "@tidal/shared";

/**
 * Largest-Triangle-Three-Buckets (LTTB) downsampling for client-side use.
 * Preserves visual peaks/valleys better than uniform stride sampling.
 */
export function lttbDownsample(
  data: MetricPoint[],
  threshold: number,
  yKey: string = "Losses/Total",
): MetricPoint[] {
  if (threshold >= data.length || threshold < 3) return data;

  const out: MetricPoint[] = [data[0]];
  const bucketSize = (data.length - 2) / (threshold - 2);

  let prevIndex = 0;

  for (let i = 1; i < threshold - 1; i++) {
    const bucketStart = Math.floor((i - 1) * bucketSize) + 1;
    const bucketEnd = Math.min(
      Math.floor(i * bucketSize) + 1,
      data.length - 1,
    );

    // Average of next bucket for the triangle target
    const nextBucketStart = Math.floor(i * bucketSize) + 1;
    const nextBucketEnd = Math.min(
      Math.floor((i + 1) * bucketSize) + 1,
      data.length - 1,
    );

    let avgX = 0;
    let avgY = 0;
    let count = 0;
    for (let j = nextBucketStart; j < nextBucketEnd; j++) {
      avgX += data[j].step;
      avgY += (data[j][yKey] as number) ?? 0;
      count++;
    }
    if (count > 0) {
      avgX /= count;
      avgY /= count;
    }

    // Find point in current bucket with max triangle area
    let maxArea = -1;
    let maxIndex = bucketStart;

    const prevX = data[prevIndex].step;
    const prevY = (data[prevIndex][yKey] as number) ?? 0;

    for (let j = bucketStart; j < bucketEnd; j++) {
      const area = Math.abs(
        (prevX - avgX) * (((data[j][yKey] as number) ?? 0) - prevY) -
          (prevX - data[j].step) * (avgY - prevY),
      );
      if (area > maxArea) {
        maxArea = area;
        maxIndex = j;
      }
    }

    out.push(data[maxIndex]);
    prevIndex = maxIndex;
  }

  out.push(data[data.length - 1]);
  return out;
}
