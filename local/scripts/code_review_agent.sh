#!/usr/bin/env bash
# code_review_agent.sh — Launch a background code review for a PR or branch.
#
# Uses the plan-review CLI directly (no Claude agent orchestrator).
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
OUTPUT_LOG="$REVIEW_DIR/.agent-output.log"
REVIEW_OUTPUT="$REVIEW_DIR/.review-output.md"
STARTED="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# Load env file if it exists
ENV_FILE="$REPO_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  set -a
  source "$ENV_FILE" 2>/dev/null || true
  set +a
fi

# ---------------------------------------------------------------------------
# Create review directory
# ---------------------------------------------------------------------------

mkdir -p "$REVIEW_DIR"

# ---------------------------------------------------------------------------
# Determine target type
# ---------------------------------------------------------------------------

if [[ "$TARGET" =~ ^[0-9]+$ ]]; then
  TARGET_TYPE="pr"
else
  TARGET_TYPE="branch"
fi

# ---------------------------------------------------------------------------
# Write runner script
# ---------------------------------------------------------------------------

RUNNER_FILE="$REVIEW_DIR/.review-runner.sh"

cat > "$RUNNER_FILE" << RUNNER_EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$REPO_DIR"

EXIT_CODE=0

if [ "$TARGET_TYPE" = "pr" ]; then
  node "$REPO_DIR/dashboard/packages/plan-review/dist/cli.js" \\
    review-pr "$TARGET" \\
    --output "$REVIEW_OUTPUT" \\
    --post-comment \\
    --budget standard \\
    > "$OUTPUT_LOG" 2>&1 || EXIT_CODE=\$?
else
  # For branch reviews, create a temp diff file and use review-plan with context
  DIFF_FILE="\$(mktemp)"
  git diff main...$TARGET > "\$DIFF_FILE" 2>/dev/null || true

  node "$REPO_DIR/dashboard/packages/plan-review/dist/cli.js" \\
    review-pr "$TARGET" \\
    --output "$REVIEW_OUTPUT" \\
    --budget standard \\
    > "$OUTPUT_LOG" 2>&1 || EXIT_CODE=\$?

  rm -f "\$DIFF_FILE"
fi

FINISHED=\$(date -u +"%Y-%m-%dT%H:%M:%SZ")
if [ "\$EXIT_CODE" -eq 0 ]; then
  FINAL_STATUS="done"
else
  FINAL_STATUS="failed"
fi

python3 -c "
import json
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
    "status": "pending",
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
  echo "Review launched in tmux session: $TMUX_SESSION"
  echo "  Monitor: tmux attach -t $TMUX_SESSION"
  echo "  Log:     tail -f $OUTPUT_LOG"
  echo "  Output:  $REVIEW_OUTPUT"
else
  echo "tmux not found — running inline"
  bash "$RUNNER_FILE"
fi
