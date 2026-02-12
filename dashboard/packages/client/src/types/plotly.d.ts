interface PlotlyModule {
  newPlot(
    root: HTMLElement,
    data: Array<Record<string, unknown>>,
    layout?: Record<string, unknown>,
    config?: Record<string, unknown>,
  ): Promise<void>;
  react(
    root: HTMLElement,
    data: Array<Record<string, unknown>>,
    layout?: Record<string, unknown>,
    config?: Record<string, unknown>,
  ): Promise<void>;
  purge(root: HTMLElement): void;
}

declare global {
  interface Window {
    Plotly?: PlotlyModule;
  }
}

export {};
