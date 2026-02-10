import { useRef, useEffect } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

interface UPlotChartProps {
  data: uPlot.AlignedData;
  options: Omit<uPlot.Options, "width" | "height">;
  height?: number;
}

export default function UPlotChart({
  data,
  options,
  height = 300,
}: UPlotChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const opts: uPlot.Options = {
      ...options,
      width,
      height,
    };

    chartRef.current = new uPlot(opts, data, containerRef.current);

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chartRef.current?.setSize({
          width: entry.contentRect.width,
          height,
        });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chartRef.current?.destroy();
      chartRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update data on changes
  useEffect(() => {
    if (chartRef.current && data[0].length > 0) {
      chartRef.current.setData(data);
    }
  }, [data]);

  return <div ref={containerRef} />;
}
