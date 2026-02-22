---
name: check-agents
description: Check status of running/completed worktree agents and code review agents. Use when the user mentions "/check-agents" or asks about agent status.
---

# Check Agent Status

Scan for running and completed implementation and code review agents and present a summary.

## Steps

### 1. Scan for Agent Status Files

Use Glob to find `.agent-status` files in:
- `.claude/worktrees/*/` — implementation agents (launched by `/implement`)
- `.claude/reviews/*/` — review agents (launched by `/code-review`)

If no status files are found in either location, tell the user: "No agents found. Use `/implement` after approving a plan to launch an implementation agent, or `/code-review` to launch a review agent."

### 1b. Read Status Files

For each `.agent-status` file found, read it and parse the JSON. Extract:
- `type`: "impl" or "review" (may be absent in older impl agents — treat as "impl")
- `status`: running, done, failed, or pending
- `branch` (impl) or `target` (review): identifier for the work
- `tmux_session`: the tmux session name
- `started`: when the agent started
- `finished`: when it completed (if done/failed)
- `exit_code`: process exit code (if done/failed)
- `commits` (impl): commit summary
- `review_dir` (review): path to the review output directory

### 2. Present Summary Table

Display a markdown table with all agents:

```
| Type   | Target / Branch  | Status  | Started              | Finished             | Details         |
|--------|-----------------|---------|----------------------|----------------------|-----------------|
| impl   | impl/my-feature | done    | 2026-02-22T10:00:00Z | 2026-02-22T10:15:00Z | 3 commits       |
| review | 42              | running | 2026-02-22T10:20:00Z | —                    | —               |
| impl   | impl/other      | running | 2026-02-22T10:25:00Z | —                    | —               |
```

### 3. Suggest Next Actions

For each agent, suggest the appropriate next step:

**Implementation agents (type: impl)**:
- **running**: "Attach with `tmux attach -t <tmux_session>` to watch live, or monitor with `tail -f <worktree>/.agent-output.log`"
- **done**: "Review the branch, create a PR with `/commit-push-pr`, then clean up the worktree with `git worktree remove <path>`"
- **failed**: "Check the log at `<worktree>/.agent-output.log` for details. You can retry with `/implement` after fixing the issue."

**Review agents (type: review)**:
- **running**: "Attach with `tmux attach -t <tmux_session>` to watch live, or monitor with `tail -f <review_dir>/.agent-output.log`"
- **done**: "Review the report at `<review_dir>/.review-output.md`. If it was a PR review, the comment was posted automatically."
- **failed**: "Check the log at `<review_dir>/.agent-output.log` for details. You can retry with `/code-review <target>`."
- **pending**: "The agent hasn't started yet. Check that tmux is running: `tmux ls`"
