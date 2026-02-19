import { describe, it, expect } from "vitest";
import { exportToMarkdown, exportToHTML } from "./reportExport.js";

// ---------------------------------------------------------------------------
// Fixtures — use plain objects cast to any to avoid strict BlockNote types
// ---------------------------------------------------------------------------

const headingBlock = {
  id: "h1",
  type: "heading",
  props: { level: 1 },
  content: [{ type: "text", text: "Report Title" }],
  children: [],
} as any;

const paragraphBlock = {
  id: "p1",
  type: "paragraph",
  props: {},
  content: [{ type: "text", text: "Some text here" }],
  children: [],
} as any;

const experimentChartBlock = {
  id: "ec1",
  type: "experimentChart",
  props: { experimentId: "exp-123", chartMode: "lm", metricKey: "Losses/Total" },
  content: [],
  children: [],
} as any;

const trajectoryBlock = {
  id: "tc1",
  type: "trajectoryChart",
  props: { experimentId: "exp-123", gatingMode: "fixed", prompt: "Once upon", analysisId: "a1" },
  content: [],
  children: [],
} as any;

const crossPromptBlock = {
  id: "cp1",
  type: "crossPromptAnalysis",
  props: { experimentId: "exp-123", gatingMode: "fixed", analysisId: "a2" },
  content: [],
  children: [],
} as any;

const sweepBlock = {
  id: "sw1",
  type: "sweepAnalysis",
  props: { experimentId: "exp-123", prompt: "Once upon", analysisId: "a3" },
  content: [],
  children: [],
} as any;

const blocks = [headingBlock, paragraphBlock, experimentChartBlock, trajectoryBlock, crossPromptBlock, sweepBlock];

// ---------------------------------------------------------------------------
// Markdown export
// ---------------------------------------------------------------------------

describe("exportToMarkdown", () => {
  it("produces valid markdown structure", () => {
    const md = exportToMarkdown("Test Report", blocks);
    expect(md).toContain("# Test Report");
    expect(md).toContain("## Report Title");
    expect(md).toContain("Some text here");
  });

  it("includes metadata placeholders for custom blocks without captures", () => {
    const md = exportToMarkdown("Test", blocks);
    expect(md).toContain("> **Chart** (LM)");
    expect(md).toContain("> **Gate Trajectory**");
    expect(md).toContain("> **Cross-Prompt Analysis**");
    expect(md).toContain("> **Gate Sweep**");
  });

  it("with captures, produces inline base64 images for chart blocks", () => {
    const captures = new Map<string, string>();
    captures.set("ec1", "data:image/png;base64,AAAA");
    captures.set("tc1", "data:image/png;base64,BBBB");

    const md = exportToMarkdown("Test", blocks, captures);
    // Chart blocks with captures get image syntax
    expect(md).toContain("![Chart (LM): exp-123](data:image/png;base64,AAAA)");
    expect(md).toContain("![Gate Trajectory: exp-123](data:image/png;base64,BBBB)");
    // Blocks without captures still get placeholders
    expect(md).toContain("> **Cross-Prompt Analysis**");
    expect(md).toContain("> **Gate Sweep**");
  });

  it("preserves current placeholder behavior when no captures provided", () => {
    const mdNoCaptures = exportToMarkdown("Test", [experimentChartBlock]);
    expect(mdNoCaptures).toContain("> **Chart** (LM)");
    expect(mdNoCaptures).not.toContain("![");
  });
});

// ---------------------------------------------------------------------------
// HTML export
// ---------------------------------------------------------------------------

describe("exportToHTML", () => {
  it("produces a self-contained HTML document", () => {
    const html = exportToHTML("Test Report", blocks);
    expect(html).toContain("<!DOCTYPE html>");
    expect(html).toContain("<title>Test Report</title>");
    expect(html).toContain("</html>");
  });

  it("includes chart placeholders for custom blocks", () => {
    const html = exportToHTML("Test", blocks);
    expect(html).toContain("chart-placeholder");
    expect(html).toContain("Gate Trajectory");
    expect(html).toContain("Cross-Prompt Analysis");
    expect(html).toContain("Gate Sweep");
  });

  it("interactive HTML includes script data blocks for custom block types", () => {
    const analysisDataMap = new Map<string, Record<string, unknown>>();
    analysisDataMap.set("tc1", { trajectory: { gateSignals: [[0.5]] } });
    analysisDataMap.set("cp1", { batchAnalysis: { perPromptSummaries: {} } });
    analysisDataMap.set("sw1", { sweepAnalysis: { interpretabilityMap: {} } });

    const html = exportToInteractiveHTML("Test", blocks, analysisDataMap);
    expect(html).toContain("<!DOCTYPE html>");
    expect(html).toContain('<script type="application/json"');
    expect(html).toContain("gateSignals");
    expect(html).toContain("batchAnalysis");
    expect(html).toContain("sweepAnalysis");
  });

  it("interactive HTML produces valid self-contained document", () => {
    const html = exportToInteractiveHTML("Test", blocks, new Map());
    expect(html).toContain("<!DOCTYPE html>");
    expect(html).toContain("</html>");
    // No CDN dependencies
    expect(html).not.toContain("cdn.");
    expect(html).not.toContain("unpkg.com");
  });
});

// Need to import the new function
import { exportToInteractiveHTML } from "./reportExport.js";
