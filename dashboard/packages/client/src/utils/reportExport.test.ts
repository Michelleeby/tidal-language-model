import { describe, it, expect } from "vitest";
import {
  exportToMarkdown,
  exportToHTML,
  exportToInteractiveHTML,
  captureBlockCanvases,
} from "./reportExport.js";

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

// ---------------------------------------------------------------------------
// XSS prevention in interactive HTML
// ---------------------------------------------------------------------------

describe("exportToInteractiveHTML — XSS prevention", () => {
  it("escapes </script> sequences in JSON data to prevent tag injection", () => {
    const maliciousData = new Map<string, Record<string, unknown>>();
    maliciousData.set("tc1", {
      trajectory: { gateSignals: [[0.5]] },
      text: '</script><script>alert("xss")</script>',
    });

    const html = exportToInteractiveHTML("Test", [trajectoryBlock], maliciousData);
    // Literal </script> inside the JSON data block must be escaped
    // The HTML should NOT contain an unescaped </script> that prematurely terminates the data tag
    const dataScriptMatch = html.match(
      /<script type="application\/json" id="data-tc1">([\s\S]*?)<\/script>/,
    );
    expect(dataScriptMatch).toBeTruthy();
    // The JSON content should have </  escaped as <\/ so the HTML parser doesn't terminate
    const jsonContent = dataScriptMatch![1];
    expect(jsonContent).not.toContain("</script>");
    expect(jsonContent).toContain("<\\/script>");
  });

  it("round-trips data correctly after escaping", () => {
    const originalData = {
      trajectory: { gateSignals: [[0.5]] },
      text: '</script><img src=x onerror=alert(1)>',
    };
    const dataMap = new Map<string, Record<string, unknown>>();
    dataMap.set("tc1", originalData);

    const html = exportToInteractiveHTML("Test", [trajectoryBlock], dataMap);
    const dataScriptMatch = html.match(
      /<script type="application\/json" id="data-tc1">([\s\S]*?)<\/script>/,
    );
    // After unescaping <\/ back to </, JSON.parse should recover original data
    const escaped = dataScriptMatch![1];
    const recovered = JSON.parse(escaped.replace(/<\\\//g, "</"));
    expect(recovered.text).toBe(originalData.text);
  });
});

// ---------------------------------------------------------------------------
// Divide-by-zero handling in chart script
// ---------------------------------------------------------------------------

describe("exportToInteractiveHTML — chart script safety", () => {
  it("chart script guards against single-element arrays", () => {
    const html = exportToInteractiveHTML("Test", [trajectoryBlock], new Map([
      ["tc1", { trajectory: { gateSignals: [[0.5]] } }],
    ]));
    // The script should handle the case where values.length or signals.length is 1
    // by avoiding division by 0 (length-1 == 0)
    expect(html).toContain("values.length <= 1");
    expect(html).toContain("signals.length <= 1");
  });
});

// ---------------------------------------------------------------------------
// captureBlockCanvases
// ---------------------------------------------------------------------------

describe("captureBlockCanvases", () => {
  it("captures canvas from blocks using closest ancestor for block ID", () => {
    // Mock BlockNote DOM: data-content-type is on .bn-block-content,
    // but data-id is on the ancestor .bn-block-outer
    const mockCanvas = {
      toDataURL: () => "data:image/png;base64,TEST123",
    };
    const mockAncestor = {
      getAttribute: (attr: string) => (attr === "data-id" ? "block-1" : null),
    };
    const mockBlockContent = {
      getAttribute: () => null,
      closest: (sel: string) => (sel === "[data-id]" ? mockAncestor : null),
      querySelector: (sel: string) => (sel === "canvas" ? mockCanvas : null),
    };
    const mockContainer = {
      querySelectorAll: (sel: string) =>
        sel === "[data-content-type]" ? [mockBlockContent] : [],
    } as unknown as HTMLElement;

    const result = captureBlockCanvases(mockContainer);
    expect(result.size).toBe(1);
    expect(result.get("block-1")).toBe("data:image/png;base64,TEST123");
  });

  it("returns empty map when container has no block elements", () => {
    const mockContainer = {
      querySelectorAll: () => [],
    } as unknown as HTMLElement;

    const result = captureBlockCanvases(mockContainer);
    expect(result.size).toBe(0);
  });

  it("skips blocks without data-id ancestor", () => {
    const mockBlockContent = {
      getAttribute: () => null,
      closest: () => null,
      querySelector: () => ({ toDataURL: () => "data:image/png;base64,TEST" }),
    };
    const mockContainer = {
      querySelectorAll: () => [mockBlockContent],
    } as unknown as HTMLElement;

    const result = captureBlockCanvases(mockContainer);
    expect(result.size).toBe(0);
  });

  it("skips blocks without canvas elements", () => {
    const mockAncestor = {
      getAttribute: (attr: string) => (attr === "data-id" ? "block-1" : null),
    };
    const mockBlockContent = {
      getAttribute: () => null,
      closest: (sel: string) => (sel === "[data-id]" ? mockAncestor : null),
      querySelector: () => null, // no canvas
    };
    const mockContainer = {
      querySelectorAll: () => [mockBlockContent],
    } as unknown as HTMLElement;

    const result = captureBlockCanvases(mockContainer);
    expect(result.size).toBe(0);
  });

  it("skips canvases that throw on toDataURL (tainted)", () => {
    const mockCanvas = {
      toDataURL: () => {
        throw new DOMException("tainted");
      },
    };
    const mockAncestor = {
      getAttribute: (attr: string) => (attr === "data-id" ? "block-1" : null),
    };
    const mockBlockContent = {
      getAttribute: () => null,
      closest: (sel: string) => (sel === "[data-id]" ? mockAncestor : null),
      querySelector: (sel: string) => (sel === "canvas" ? mockCanvas : null),
    };
    const mockContainer = {
      querySelectorAll: () => [mockBlockContent],
    } as unknown as HTMLElement;

    const result = captureBlockCanvases(mockContainer);
    expect(result.size).toBe(0);
  });

  it("captures multiple blocks", () => {
    function makeBlock(id: string, dataUrl: string) {
      return {
        getAttribute: () => null,
        closest: (sel: string) =>
          sel === "[data-id]"
            ? { getAttribute: (a: string) => (a === "data-id" ? id : null) }
            : null,
        querySelector: (sel: string) =>
          sel === "canvas" ? { toDataURL: () => dataUrl } : null,
      };
    }
    const mockContainer = {
      querySelectorAll: () => [
        makeBlock("b1", "data:image/png;base64,AAA"),
        makeBlock("b2", "data:image/png;base64,BBB"),
      ],
    } as unknown as HTMLElement;

    const result = captureBlockCanvases(mockContainer);
    expect(result.size).toBe(2);
    expect(result.get("b1")).toBe("data:image/png;base64,AAA");
    expect(result.get("b2")).toBe("data:image/png;base64,BBB");
  });
});
