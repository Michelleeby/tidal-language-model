import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { parseADR, summarizeADRs } from "../adr/adr-summarizer.js";
import { selectRelevantADRs } from "../adr/adr-relevance.js";
import type { ADRSummary } from "../types.js";

// ---------------------------------------------------------------------------
// Sample ADR content for testing
// ---------------------------------------------------------------------------

const SAMPLE_ADR = `# 0001. Single Modulation Gate

**Date:** 2026-01-15
**Status:** Accepted

## Context

The original 3-gate system (temperature, top-k, top-p) demonstrated that
redundant action dimensions collapse to constants. The PPO agent proved it
doesn't need separate knobs for distribution peakedness parameters.

## Decision

Reduce the action space to a single 1D modulation gate that controls a
conservative-to-exploratory axis. The \`GatingModulator\` maps this signal
to all generation parameters.

### Implementation

Changes to \`plugins/tidal/GatingPolicyAgent.py\` and
\`plugins/tidal/GatingModulator.py\`.

## Consequences

### Positive
- Simpler action space
- Faster convergence

### Negative
- Less fine-grained control

### Neutral
- Checkpoint format unchanged

## Alternatives Considered

### Multi-dimensional action space
Keep all 3 gates. Rejected due to collapse.

## References

- Code: \`plugins/tidal/GatingPolicyAgent.py\`
- Code: \`plugins/tidal/GatingModulator.py\`
- Config: \`plugins/tidal/configs/rl_config.yaml\`
`;

const SAMPLE_ADR_2 = `# 0004. Lazy Disk Cache for MCP HTTP Client

**Date:** 2026-02-10
**Status:** Accepted

## Context

The MCP server makes repeated HTTP calls to the dashboard API for the same
experiment data. Caching reduces latency and API load.

## Decision

Add a \`CachingTidalApiClient\` wrapper that lazily caches GET responses
to disk in \`~/.cache/tidal/\`.

## Consequences

### Positive
- Faster responses

### Negative
- Stale data possible

### Neutral
- Cache dir is configurable

## Alternatives Considered

### In-memory LRU
Would not persist across restarts.

## References

- Code: \`dashboard/packages/mcp/src/http-client.ts\`
`;

// ---------------------------------------------------------------------------
// parseADR
// ---------------------------------------------------------------------------

describe("parseADR", () => {
  it("extracts number and title from heading", () => {
    const result = parseADR(SAMPLE_ADR, "0001-single-modulation-gate.md");
    assert.equal(result.number, 1);
    assert.equal(result.title, "Single Modulation Gate");
  });

  it("extracts status", () => {
    const result = parseADR(SAMPLE_ADR, "0001-single-modulation-gate.md");
    assert.equal(result.status, "Accepted");
  });

  it("extracts context paragraph", () => {
    const result = parseADR(SAMPLE_ADR, "0001-single-modulation-gate.md");
    assert.ok(result.context.includes("3-gate system"));
    assert.ok(result.context.includes("redundant action dimensions"));
  });

  it("extracts decision paragraph", () => {
    const result = parseADR(SAMPLE_ADR, "0001-single-modulation-gate.md");
    assert.ok(result.decision.includes("single 1D modulation gate"));
  });

  it("extracts backtick-quoted keywords", () => {
    const result = parseADR(SAMPLE_ADR, "0001-single-modulation-gate.md");
    assert.ok(result.keywords.includes("GatingModulator"));
    assert.ok(result.keywords.includes("GatingPolicyAgent"));
  });

  it("extracts files affected from references", () => {
    const result = parseADR(SAMPLE_ADR, "0001-single-modulation-gate.md");
    assert.ok(result.files_affected.includes("plugins/tidal/GatingPolicyAgent.py"));
    assert.ok(result.files_affected.includes("plugins/tidal/GatingModulator.py"));
    assert.ok(result.files_affected.includes("plugins/tidal/configs/rl_config.yaml"));
  });

  it("builds raw summary string", () => {
    const result = parseADR(SAMPLE_ADR, "0001-single-modulation-gate.md");
    assert.ok(result.raw.startsWith("ADR 1: Single Modulation Gate (Accepted)."));
    assert.ok(result.raw.includes("Decision:"));
  });

  it("handles ADR without references section", () => {
    const minimal = `# 0099. Minimal ADR

**Date:** 2026-01-01
**Status:** Proposed

## Context

Some context here.

## Decision

Some decision here.
`;
    const result = parseADR(minimal, "0099-minimal.md");
    assert.equal(result.number, 99);
    assert.equal(result.title, "Minimal ADR");
    assert.equal(result.files_affected.length, 0);
  });
});

// ---------------------------------------------------------------------------
// summarizeADRs
// ---------------------------------------------------------------------------

describe("summarizeADRs", () => {
  it("parses multiple ADR contents into summaries", () => {
    const entries = [
      { filename: "0001-single-modulation-gate.md", content: SAMPLE_ADR },
      { filename: "0004-lazy-disk-cache.md", content: SAMPLE_ADR_2 },
    ];
    const results = summarizeADRs(entries);
    assert.equal(results.length, 2);
    assert.equal(results[0].number, 1);
    assert.equal(results[1].number, 4);
  });

  it("sorts by ADR number", () => {
    const entries = [
      { filename: "0004-lazy-disk-cache.md", content: SAMPLE_ADR_2 },
      { filename: "0001-single-modulation-gate.md", content: SAMPLE_ADR },
    ];
    const results = summarizeADRs(entries);
    assert.equal(results[0].number, 1);
    assert.equal(results[1].number, 4);
  });
});

// ---------------------------------------------------------------------------
// selectRelevantADRs
// ---------------------------------------------------------------------------

describe("selectRelevantADRs", () => {
  const summaries: ADRSummary[] = [
    parseADR(SAMPLE_ADR, "0001-single-modulation-gate.md"),
    parseADR(SAMPLE_ADR_2, "0004-lazy-disk-cache.md"),
  ];

  it("scores ADRs by keyword overlap with plan text", () => {
    const plan = "Modify GatingModulator to add new parameter mapping";
    const selected = selectRelevantADRs(summaries, plan, 5000);
    assert.ok(selected.length >= 1);
    assert.equal(selected[0].number, 1); // GatingModulator is in ADR 1
  });

  it("includes ADRs whose files overlap with plan-mentioned files", () => {
    const plan = "Changes to plugins/tidal/GatingPolicyAgent.py";
    const selected = selectRelevantADRs(summaries, plan, 5000);
    assert.ok(selected.some((s) => s.number === 1));
  });

  it("always includes the most recent ADR", () => {
    const plan = "Something completely unrelated to any ADR";
    const selected = selectRelevantADRs(summaries, plan, 5000);
    assert.ok(selected.some((s) => s.number === 4)); // highest number = most recent
  });

  it("respects token budget", () => {
    const plan = "GatingModulator GatingPolicyAgent CachingTidalApiClient";
    // Budget large enough for 1 ADR but not both
    const adr1Tokens = Math.ceil(summaries[0].raw.length / 4);
    const adr2Tokens = Math.ceil(summaries[1].raw.length / 4);
    const budget = Math.max(adr1Tokens, adr2Tokens) + 10; // room for 1, not both
    const selected = selectRelevantADRs(summaries, plan, budget);
    assert.ok(selected.length >= 1);
    assert.ok(selected.length < summaries.length);
  });
});
