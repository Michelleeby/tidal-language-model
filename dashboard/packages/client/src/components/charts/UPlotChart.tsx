import { useRef, useEffect, useImperativeHandle, forwardRef, useCallback } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

export interface UPlotChartHandle {
  resetZoom: () => void;
}

interface UPlotChartProps {
  data: uPlot.AlignedData;
  options: Omit<uPlot.Options, "width" | "height">;
  height?: number;
  onZoomChange?: (zoomed: boolean) => void;
}

export default forwardRef<UPlotChartHandle, UPlotChartProps>(
  function UPlotChart({ data, options, height = 300, onZoomChange }, ref) {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<uPlot | null>(null);
    const zoomedRef = useRef(false);

    const resetZoom = useCallback(() => {
      if (chartRef.current) {
        const xData = chartRef.current.data[0];
        if (xData.length > 0) {
          chartRef.current.setScale("x", {
            min: xData[0],
            max: xData[xData.length - 1],
          });
        }
        zoomedRef.current = false;
        onZoomChange?.(false);
      }
    }, [onZoomChange]);

    useImperativeHandle(ref, () => ({ resetZoom }), [resetZoom]);

    useEffect(() => {
      if (!containerRef.current) return;

      const width = containerRef.current.clientWidth;
      const opts: uPlot.Options = {
        ...options,
        width,
        height,
        cursor: {
          ...options.cursor,
          drag: { x: true, y: false, ...(options.cursor as any)?.drag },
        },
        hooks: {
          ...options.hooks,
          setScale: [
            ...(options.hooks?.setScale ?? []),
            (u: uPlot, scaleKey: string) => {
              if (scaleKey !== "x") return;
              const xData = u.data[0];
              if (xData.length === 0) return;

              const scaleMin = u.scales.x.min!;
              const scaleMax = u.scales.x.max!;
              const dataMin = xData[0];
              const dataMax = xData[xData.length - 1];

              // Consider zoomed if the visible range is meaningfully smaller than the full range
              const isZoomed =
                scaleMin > dataMin + 1 || scaleMax < dataMax - 1;

              if (isZoomed !== zoomedRef.current) {
                zoomedRef.current = isZoomed;
                onZoomChange?.(isZoomed);
              }
            },
          ],
        },
      };

      chartRef.current = new uPlot(opts, data, containerRef.current);

      // Double-click to reset zoom
      const el = containerRef.current;
      const handleDblClick = () => resetZoom();
      el.addEventListener("dblclick", handleDblClick);

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
        el.removeEventListener("dblclick", handleDblClick);
        ro.disconnect();
        chartRef.current?.destroy();
        chartRef.current = null;
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Update data on changes — preserve scale when zoomed
    useEffect(() => {
      if (chartRef.current && data[0].length > 0) {
        chartRef.current.setData(data, !zoomedRef.current);
      }
    }, [data]);

    return <div ref={containerRef} />;
  },
);
