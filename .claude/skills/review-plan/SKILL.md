---
name: review-plan
description: Review an implementation plan using multiple AI models (GPT-4o, Gemini Flash) across 5 dimensions. Use when the user mentions "/review-plan", asks to review a plan, or after creating an implementation plan that would benefit from external validation.
---

# Plan Review

Send an implementation plan to external AI models for structured review across 5 dimensions: completeness, blind spots, regression risk, test coverage, and hypothesis scope. Uses OpenAI (GPT-4o) and Google (Gemini Flash) only — no Anthropic tokens consumed. Feedback is deduplicated and ranked by severity.

## 1. Identify the Plan

- If the user provides plan text directly, use that.
- If a plan was just created in the current session (via plan mode), use that plan.
- Otherwise, check `research/plans/` for the most recent plan file.
- If no plan is found, ask the user to provide one.

## 2. Gather Context

Read files mentioned in the plan to provide additional context to reviewers:
- Read up to 5 files referenced in the plan (max 200 lines each)
- Concatenate their contents as supplementary context

## 3. Check Provider Availability

Call `list_review_providers` to see which API keys are configured. Report to the user:
- Which providers are available (OpenAI, Google)
- If none are available, inform the user they need to set API keys

## 4. Auto-Select Budget

Based on plan size:
- **Small plan** (<50 lines): `minimal` (1 model per dimension, ~$0.01-0.03)
- **Medium plan** (50-200 lines): `standard` (2 models per dimension, ~$0.03-0.08)
- **Large plan** (>200 lines): `thorough` (3 model calls per dimension using gpt-4o, o3-mini, gemini-flash, ~$0.08-0.20)

## 5. Auto-Select Dimensions

Skip dimensions that don't apply:
- Skip `hypothesis_scope` for non-experiment plans (no hypothesis, no ML metrics)
- Skip `regression_risk` for plans that only create new files (no existing code changes)
- Always include `completeness`, `blind_spots`, and `test_coverage`

## 6. Execute Review

Call `review_plan` with:
- `plan`: the full plan text
- `dimensions`: the selected dimensions
- `context`: gathered file contents
- `include_adrs`: true (always inject relevant ADR context)
- `budget`: the auto-selected budget tier

## 7. Present Results

Format the output as a structured report:

### Critical Issues
List any items with severity "critical" — these should be addressed before implementation.

### Warnings
List items with severity "warning" — these are worth considering but may not block progress.

### Suggestions
List items with severity "suggestion" — nice-to-have improvements.

### Summary
- Number of items found per dimension
- Which providers contributed feedback
- Overall assessment: "Plan looks solid" / "Some concerns to address" / "Significant gaps identified"

For each item, show:
- **Description**: What the issue is
- **Affected files**: Which files are impacted
- **Reasoning**: Why this matters
- **Corroborated by**: Which models flagged this (items flagged by multiple models are more likely real issues)

## 8. Follow-Up

Ask the user what they'd like to do:
1. **Address issues** — update the plan to fix critical/warning items
2. **Deeper review** — re-run on a specific dimension with all providers
3. **Proceed as-is** — accept the plan and begin implementation

## 9. Wrap-Up

1. Save the plan to the research/plans/ directory like f"{timestamp}_{title_slug}.md" where title slug is a descriptive name from the content of the plan.
