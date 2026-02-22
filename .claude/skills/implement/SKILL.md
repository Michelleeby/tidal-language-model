---
name: implement
description: Launch a background worktree agent to implement an approved plan. Use after ExitPlanMode with an approved plan, or when the user mentions "/implement".
---

# Implement Plan in Worktree Agent

After plan approval, delegate implementation to an isolated background agent instead of coding directly in this session.

## Steps

### 1. Find the Most Recent Plan

Use Glob to find `.md` files in `~/.claude/plans/`. Read the directory and identify the most recently modified plan file.

If no plan files exist, tell the user: "No plan files found in ~/.claude/plans/. Please create and approve a plan first."

### 2. Read the Plan and Extract Title

Read the plan file. Extract the title from the first `# ` heading line.

If the plan file is empty or has no heading, tell the user and stop.

### 3. Derive Branch Name

From the extracted title:

1. Strip a leading `Plan:` or `Plan -` prefix if present (case-insensitive), then trim whitespace.
2. Convert to kebab-case: lowercase, replace spaces and underscores with hyphens, strip characters that aren't alphanumeric or hyphens, collapse consecutive hyphens, trim leading/trailing hyphens.
3. Truncate to 60 characters (at a hyphen boundary if possible).
4. Scan the full plan content for ADR references matching the pattern `ADR[- ]?(\d{4})` (case-insensitive). If found, prefix the branch name with `adr-NNNN/` using the first match. Otherwise, prefix with `impl/`.

Examples:
- "Plan: Worktree Agent System" → `impl/worktree-agent-system`
- "Input-Dependent Gating (ADR 0008)" → `adr-0008/input-dependent-gating`

### 4. Launch the Worktree Agent

Run the following command using Bash:

```
bash local/scripts/worktree_agent.sh "<branch-name>" "<plan-file-path>"
```

### 5. Report to User

After the script runs, parse its stdout output (key=value pairs) and report:

- **Branch**: the branch name
- **Worktree**: the worktree directory path
- **tmux session**: the session name (from `TMUX_SESSION=` output)
- **Status**: agent is running in a tmux session
- **Log file**: path to `.agent-output.log` for live monitoring
- **Next steps**: use `/check-agents` to monitor progress, `tmux attach -t <session>` to watch live, or `tail -f <log-file>` to stream the log

### 6. Do NOT Start Coding

The agent handles all implementation. Stay available for conversation, ADRs, and launching more agents.
