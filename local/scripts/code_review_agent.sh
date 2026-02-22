#!/usr/bin/env bash
# code_review_agent.sh — Launch a background code review agent for a PR or branch.
#
# Usage: code_review_agent.sh <target> [--dry-run]
#   target: PR number (e.g. "42") or branch name (e.g. "impl/my-feature")
#   --dry-run: Create review dir and status file, but do not launch tmux session

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

if [ $# -lt 1 ]; then
  echo "Usage: $0 <target> [--dry-run]"
  echo "  target: PR number or branch name to review"
  echo "  --dry-run: set up files without launching tmux"
  exit 1
fi

TARGET="$1"
DRY_RUN=false
if [ "${2:-}" = "--dry-run" ]; then
  DRY_RUN=true
fi

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")")"
REVIEW_BASE="$REPO_DIR/.claude/reviews"
REVIEW_DIR="$REVIEW_BASE/$TARGET"
TMUX_SESSION="review-$TARGET"
STATUS_FILE="$REVIEW_DIR/.agent-status"
PROMPT_FILE="$REVIEW_DIR/.agent-prompt.txt"
RUNNER_FILE="$REVIEW_DIR/.agent-runner.sh"
OUTPUT_LOG="$REVIEW_DIR/.agent-output.log"
STARTED="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
MCP_CONFIG_FILE="$REPO_DIR/.mcp.json"
ENV_FILE="$REPO_DIR/.env"

# ---------------------------------------------------------------------------
# Create review directory
# ---------------------------------------------------------------------------

mkdir -p "$REVIEW_DIR"

# ---------------------------------------------------------------------------
# Write agent prompt
# ---------------------------------------------------------------------------

# Determine if target looks like a PR number (all digits) or a branch name
if [[ "$TARGET" =~ ^[0-9]+$ ]]; then
  DIFF_CMD="gh pr diff $TARGET"
  POST_CMD="gh pr comment $TARGET --body-file .claude/reviews/$TARGET/.review-output.md"
  TARGET_TYPE="pr"
else
  DIFF_CMD="git diff main...$TARGET"
  POST_CMD=""
  TARGET_TYPE="branch"
fi

cat > "$PROMPT_FILE" << PROMPT_EOF
You are a code review agent. Your job is to review changes and produce a structured review report.

## Target

Review target: $TARGET (type: $TARGET_TYPE)

## Steps

1. Get the diff:
   - If PR: \`$DIFF_CMD\`
   - If branch: \`$DIFF_CMD\`

2. Read the contents of up to 5 most-changed files for context.

3. Call \`list_review_providers\` (from the plan-review MCP) to confirm which providers are available.

4. Call \`summarize_adrs\` to get relevant ADR context. Use keywords from the diff.

5. Call \`review_code\` with:
   - diff: the full diff output
   - context: relevant file contents joined together
   - include_adrs: true
   - budget: "standard"

6. Format the results as a markdown report with:
   - ## Summary (1-2 sentences)
   - ## Critical Issues (if any)
   - ## Warnings
   - ## Suggestions
   - ## ADR Compliance
   - Each item: severity badge, description, affected files, reasoning

7. Write the report to: .claude/reviews/$TARGET/.review-output.md

8. If target is a PR number, post the report as a PR comment:
   \`gh pr comment $TARGET --body-file .claude/reviews/$TARGET/.review-output.md\`

## Important

- This is a READ-ONLY review. Do not modify any source files.
- Write output ONLY to .claude/reviews/$TARGET/
- If any step fails gracefully, continue with available information.
PROMPT_EOF

# ---------------------------------------------------------------------------
# Build MCP config with env var substitution
# ---------------------------------------------------------------------------

MCP_CONFIG='{}'
if [ -f "$MCP_CONFIG_FILE" ]; then
  # Load env file if it exists
  if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    set -a
    source "$ENV_FILE" 2>/dev/null || true
    set +a
  fi

  # Inline the plan-review MCP server config
  OPENAI_KEY="${OPENAI_API_KEY:-}"
  GOOGLE_KEY="${GOOGLE_AI_API_KEY:-}"
  ANTHROPIC_KEY="${ANTHROPIC_API_KEY:-}"
  ADR_DIR="${TIDAL_ADR_DIR:-}"

  MCP_CONFIG=$(python3 - <<PYTHON_EOF
import json
config = {
    "mcpServers": {
        "plan-review": {
            "command": "node",
            "args": ["dashboard/packages/plan-review/dist/index.js"],
            "env": {
                "OPENAI_API_KEY": "$OPENAI_KEY",
                "GOOGLE_AI_API_KEY": "$GOOGLE_KEY",
                "ANTHROPIC_API_KEY": "$ANTHROPIC_KEY",
                "TIDAL_ADR_DIR": "$ADR_DIR"
            }
        }
    }
}
print(json.dumps(config))
PYTHON_EOF
)
fi

# ---------------------------------------------------------------------------
# Write agent runner script
# ---------------------------------------------------------------------------

cat > "$RUNNER_FILE" << RUNNER_EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$REPO_DIR"

# Unset CLAUDECODE to avoid nested session guard
unset CLAUDECODE

EXIT_CODE=0
claude --model sonnet -p "\$(cat '$PROMPT_FILE')" \\
  --allowedTools "Read" "Glob" "Grep" \\
    "Bash(gh pr diff:*)" "Bash(gh pr view:*)" "Bash(gh pr comment:*)" \\
    "Bash(git diff:*)" "Bash(git log:*)" "Bash(git show:*)" "Bash(git status:*)" \\
    "Bash(ls:*)" "Bash(head:*)" "Bash(tail:*)" "Bash(wc:*)" \\
    "Write(.claude/reviews/*)" \\
  --no-session-persistence \\
  --disable-slash-commands \\
  --strict-mcp-config --mcp-config '$MCP_CONFIG' \\
  > '$OUTPUT_LOG' 2>&1 || EXIT_CODE=\$?

FINISHED=\$(date -u +"%Y-%m-%dT%H:%M:%SZ")
if [ "\$EXIT_CODE" -eq 0 ]; then
  FINAL_STATUS="done"
else
  FINAL_STATUS="failed"
fi

python3 -c "
import json, sys
status = {
    'type': 'review',
    'status': '\$FINAL_STATUS',
    'target': '$TARGET',
    'started': '$STARTED',
    'finished': '\$FINISHED',
    'exit_code': \$EXIT_CODE,
    'review_dir': '$REVIEW_DIR',
    'tmux_session': '$TMUX_SESSION'
}
with open('$STATUS_FILE', 'w') as f:
    json.dump(status, f, indent=2)
"
RUNNER_EOF

chmod +x "$RUNNER_FILE"

# ---------------------------------------------------------------------------
# Write initial status file
# ---------------------------------------------------------------------------

python3 - << STATUS_EOF
import json
status = {
    "type": "review",
    "status": "pending" if True else "pending",
    "target": "$TARGET",
    "started": "$STARTED",
    "review_dir": "$REVIEW_DIR",
    "tmux_session": "$TMUX_SESSION"
}
with open("$STATUS_FILE", "w") as f:
    json.dump(status, f, indent=2)
STATUS_EOF

# ---------------------------------------------------------------------------
# Launch (or skip if dry-run)
# ---------------------------------------------------------------------------

if [ "$DRY_RUN" = "true" ]; then
  echo "Dry-run: review dir created at $REVIEW_DIR"
  echo "  Status: $STATUS_FILE"
  echo "  Prompt: $PROMPT_FILE"
  echo "  Runner: $RUNNER_FILE"
  echo "  Log:    $OUTPUT_LOG (not created yet)"
  echo ""
  echo "To launch for real: bash local/scripts/code_review_agent.sh $TARGET"
  exit 0
fi

# Update status to running
python3 - << RUNNING_EOF
import json
with open("$STATUS_FILE") as f:
    status = json.load(f)
status["status"] = "running"
with open("$STATUS_FILE", "w") as f:
    json.dump(status, f, indent=2)
RUNNING_EOF

# Launch in tmux
if command -v tmux &>/dev/null; then
  tmux new-session -d -s "$TMUX_SESSION" "bash $RUNNER_FILE"
  echo "Review agent launched in tmux session: $TMUX_SESSION"
  echo "  Monitor: tmux attach -t $TMUX_SESSION"
  echo "  Log:     tail -f $OUTPUT_LOG"
  echo "  Output:  $REVIEW_DIR/.review-output.md"
else
  echo "tmux not found — running inline"
  bash "$RUNNER_FILE"
fi
