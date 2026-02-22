---
name: adr
description: Create an Architecture Decision Record (ADR). Use when the user wants to document an architecture decision, mentions "/adr", or discusses recording a technical decision for the project.
---

# Architecture Decision Record (ADR)

When the user asks to create an ADR, follow this process:

## 1. Determine the Next ADR Number

Scan the `research/adrs/` directory for existing ADRs. Files are named `{NNNN}-{slugified-title}.md` (e.g., `0001-use-ppo-for-gating-controller.md`). The next number is one higher than the highest existing number, zero-padded to 4 digits. If no ADRs exist yet, start at `0001`.

## 2. Gather Context

Before writing, review relevant code, plans, session transcripts, and existing ADRs to understand the decision's context. Ask the user clarifying questions if the decision scope is unclear.

## 3. Write the ADR

Create the file at `research/adrs/{NNNN}-{slugified-title}.md` using the template below. Create the `research/adrs/` directory if it doesn't exist.

Write thoughtful, context-aware content — not just fill-in-the-blank boilerplate. Reference specific experiments, code paths, and data where relevant.

### Template

```markdown
# {NUMBER}. {Title}

**Date:** {YYYY-MM-DD}
**Status:** {Proposed | Accepted | Deprecated | Superseded by [NNNN](link)}

## Context

{What is the problem or need? What forces are at play? Reference specific
experiments, metrics, or code that motivated this decision.}

## Decision

{Concise statement of the decision.}

{Implementation details — what specifically changes, which files/modules
are affected, and how the change integrates with the existing architecture.}

## Consequences

### Positive
- {Benefit 1}
- {Benefit 2}

### Negative
- {Tradeoff or cost 1}
- {Tradeoff or cost 2}

### Neutral
- {Side effect that is neither clearly positive nor negative}

## Alternatives Considered

### {Alternative 1 name}
{Brief description of the approach and why it was rejected.}

### {Alternative 2 name}
{Brief description of the approach and why it was rejected.}

## References

- {Related ADR: [NNNN — Title](../adrs/NNNN-title.md)}
- {Code path: `plugins/tidal/SomeModule.py`}
- {Plan: `research/plans/YYYYMMDD_title.md`}
- {Session: `research/sessions/YYYYMMDD_title.md`}
```

## 4. After Writing

- Confirm the file was created and show the user the path.
- If this ADR supersedes an existing one, update the superseded ADR's status to `Superseded by [NNNN](link)`.

## 5. Update README

Update the ADR table in `README.md` so it lists **every** ADR in `research/adrs/`, not just the one you created.

1. Scan `research/adrs/` for all ADR files.
2. For each file, extract the ADR number and title from the first `# {NUMBER}. {Title}` heading.
3. Rebuild the table rows in numeric order using this format:
   ```
   | [NNNN](research/adrs/{filename}) | {Title} |
   ```
4. Replace the existing table rows (between the `|---|---|` header separator and the next blank line) with the rebuilt rows.

This ensures the README stays in sync even if previous ADRs were added without updating it.
