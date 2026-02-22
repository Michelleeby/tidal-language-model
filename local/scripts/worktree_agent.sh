#!/usr/bin/env bash
# worktree_agent.sh — Launch a Claude Code agent in an isolated git worktree.
#
# Usage: bash local/scripts/worktree_agent.sh <branch-name> <plan-file-path> [--dry-run]
#
# Creates a worktree, copies the plan, writes a status file, and launches
# claude -p in the background with scoped tool permissions.
#
# --dry-run: Set up worktree and status file but skip launching claude.

set -euo pipefail

# --- Parse arguments ---

if [ $# -lt 2 ]; then
  echo "Usage: bash local/scripts/worktree_agent.sh <branch-name> <plan-file-path> [--dry-run]"
  exit 1
fi

BRANCH="$1"
PLAN_FILE="$2"
DRY_RUN=false

if [ "${3:-}" = "--dry-run" ]; then
  DRY_RUN=true
fi

# --- Check tmux (required for non-dry-run) ---

if [ "$DRY_RUN" != true ]; then
  if ! command -v tmux &>/dev/null; then
    echo "Error: tmux is required but not installed."
    echo "Install with: brew install tmux"
    exit 1
  fi
fi

# --- Resolve repo root ---

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Validate inputs ---

if [ ! -f "$PLAN_FILE" ]; then
  echo "Error: Plan file not found: $PLAN_FILE"
  exit 1
fi

# Check if branch already exists (local branch or worktree)
if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$BRANCH" 2>/dev/null; then
  echo "Error: Branch '$BRANCH' already exists."
  exit 1
fi

WORKTREE_DIR="$REPO_ROOT/.claude/worktrees/$BRANCH"

if [ -d "$WORKTREE_DIR" ]; then
  echo "Error: Worktree directory already exists: $WORKTREE_DIR"
  exit 1
fi

# --- Create worktree ---

mkdir -p "$(dirname "$WORKTREE_DIR")"
git -C "$REPO_ROOT" worktree add "$WORKTREE_DIR" -b "$BRANCH" HEAD

# --- Copy .env if present (needed by tests, gitignored) ---

if [ -f "$REPO_ROOT/.env" ]; then
  cp "$REPO_ROOT/.env" "$WORKTREE_DIR/.env"
fi

# --- Copy plan file ---

cp "$PLAN_FILE" "$WORKTREE_DIR/PLAN.md"

# --- Write status file ---

STATUS_FILE="$WORKTREE_DIR/.agent-status"
STARTED=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
SESSION_NAME="agent-$(echo "$BRANCH" | tr '/' '-' | tr '.:' '--')"

python3 -c "
import json, sys
status = {
    'status': 'running',
    'started': '$STARTED',
    'branch': '$BRANCH',
    'plan_file': '$(basename "$PLAN_FILE")',
    'worktree': '$WORKTREE_DIR',
    'tmux_session': '$SESSION_NAME'
}
json.dump(status, sys.stdout, indent=2)
" > "$STATUS_FILE"

LOG_FILE="$WORKTREE_DIR/.agent-output.log"

# --- Output key=value pairs ---

echo "WORKTREE=$WORKTREE_DIR"
echo "BRANCH=$BRANCH"
echo "STATUS_FILE=$STATUS_FILE"
echo "LOG_FILE=$LOG_FILE"
echo "TMUX_SESSION=$SESSION_NAME"

# --- Dry-run stops here ---

if [ "$DRY_RUN" = true ]; then
  echo "DRY_RUN=true (skipping claude launch)"
  exit 0
fi

# --- Build prompt file ---

SYSTEM_APPEND="You are running in an isolated git worktree on branch '$BRANCH'. Follow TDD strictly: write tests, commit when they fail (red), implement, commit when they pass (green). Each red/green cycle is a separate commit. Do NOT push to remote. Do NOT modify files outside this worktree."

{
  cat <<'PROMPT_HEADER'
You are an implementation agent working in an isolated git worktree.

Your task is to implement the following plan:

PROMPT_HEADER
  cat "$WORKTREE_DIR/PLAN.md"
  cat <<PROMPT_FOOTER

## Instructions

1. Read the plan carefully and understand all requirements.
2. Follow TDD strictly with separate commits for each phase:
   a. **Red**: Write tests for one unit of work. Run them and verify they FAIL.
      Commit with message: "test: <what the tests cover> (red)"
   b. **Green**: Write the minimum code to make those tests pass. Run them and verify they PASS.
      Commit with message: "feat: <what was implemented> (green)"
   c. Repeat (a) and (b) for each unit of work in the plan.
3. If the plan has a TDD Sequence section, follow that order.
4. Do NOT push — the main session will handle PR creation.
5. IMPORTANT: After your final commit, stop immediately. Do not summarize, do not review, do not run additional checks. Just stop.

## Context

- You are on branch: $BRANCH
- Working directory: $WORKTREE_DIR
- This is an isolated worktree — changes here do not affect the main working tree.
- Test commands: see CLAUDE.md in the repo root for test commands.
PROMPT_FOOTER
} > "$WORKTREE_DIR/.agent-prompt.txt"

# --- Write runner script ---

cat <<RUNNER_EOF > "$WORKTREE_DIR/.agent-runner.sh"
#!/usr/bin/env bash
set -euo pipefail

cd "$WORKTREE_DIR"

# Unset CLAUDECODE to avoid "nested session" guard when launched from Claude Code
unset CLAUDECODE

EXIT_CODE=0
claude --model sonnet -p "\$(cat .agent-prompt.txt)" \\
  --allowedTools "Edit" "Write" "Read" "Glob" "Grep" \\
    "Bash(python -m unittest:*)" "Bash(python3:*)" "Bash(python:*)" \\
    "Bash(git add:*)" "Bash(git commit:*)" "Bash(git diff:*)" \\
    "Bash(git status:*)" "Bash(git log:*)" "Bash(ls:*)" \\
    "Bash(source:*)" "Bash(npm test:*)" "Bash(npx vitest:*)" \\
    "Bash(head:*)" "Bash(tail:*)" "Bash(wc:*)" \\
  --no-session-persistence \\
  --disable-slash-commands \\
  --strict-mcp-config --mcp-config '{"mcpServers":{}}' \\
  --append-system-prompt "$SYSTEM_APPEND" \\
  > .agent-output.log 2>&1 || EXIT_CODE=\$?

# Capture commit summary
COMMITS=\$(git log HEAD --not main --oneline 2>/dev/null || echo "(none)")
FINISHED=\$(date -u +"%Y-%m-%dT%H:%M:%SZ")

if [ "\$EXIT_CODE" -eq 0 ]; then
  FINAL_STATUS="done"
else
  FINAL_STATUS="failed"
fi

python3 -c "
import json, sys
status = {
    'status': '\$FINAL_STATUS',
    'started': '$STARTED',
    'finished': '\$FINISHED',
    'branch': '$BRANCH',
    'exit_code': \$EXIT_CODE,
    'commits': '''\$COMMITS'''.strip(),
    'worktree': '$WORKTREE_DIR',
    'tmux_session': '$SESSION_NAME'
}
json.dump(status, sys.stdout, indent=2)
" > "$WORKTREE_DIR/.agent-status"
RUNNER_EOF

chmod +x "$WORKTREE_DIR/.agent-runner.sh"

# --- Launch in tmux ---

tmux new-session -d -s "$SESSION_NAME" "bash $WORKTREE_DIR/.agent-runner.sh"
