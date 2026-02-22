#!/usr/bin/env bash
# Test suite for code_review_agent.sh
# Follows the same pattern as test_worktree_agent.sh

set -euo pipefail

WORKTREE_DIR="/Users/michellebyrnes/tidal-language-model/.claude/worktrees/impl/background-multi-model-code-review-skill"
SCRIPT="$WORKTREE_DIR/local/scripts/code_review_agent.sh"
TEST_REVIEW_BASE="$WORKTREE_DIR/.claude/reviews"

PASS=0
FAIL=0

pass() { echo "PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "FAIL: $1"; FAIL=$((FAIL + 1)); }

# ---------------------------------------------------------------------------
# Test 1: Script exists and is executable
# ---------------------------------------------------------------------------
if [ -f "$SCRIPT" ] && [ -x "$SCRIPT" ]; then
  pass "script exists and is executable"
else
  fail "script exists and is executable (got: file=$([ -f "$SCRIPT" ] && echo yes || echo no), exec=$([ -x "$SCRIPT" ] && echo yes || echo no))"
fi

# ---------------------------------------------------------------------------
# Test 2: Missing arguments shows usage
# ---------------------------------------------------------------------------
usage_output=$("$SCRIPT" 2>&1 || true)
if echo "$usage_output" | grep -qi "usage"; then
  pass "missing arguments shows usage"
else
  fail "missing arguments shows usage (got: $usage_output)"
fi

# ---------------------------------------------------------------------------
# Test 3: Dry-run creates review directory
# ---------------------------------------------------------------------------
TEST_TARGET="test-pr-999"
TEST_REVIEW_DIR="$TEST_REVIEW_BASE/$TEST_TARGET"

# Clean up from previous runs
rm -rf "$TEST_REVIEW_DIR"

"$SCRIPT" "$TEST_TARGET" --dry-run

if [ -d "$TEST_REVIEW_DIR" ]; then
  pass "dry-run creates review directory"
else
  fail "dry-run creates review directory (dir not found: $TEST_REVIEW_DIR)"
fi

# ---------------------------------------------------------------------------
# Test 4: Dry-run creates .agent-status file
# ---------------------------------------------------------------------------
STATUS_FILE="$TEST_REVIEW_DIR/.agent-status"
if [ -f "$STATUS_FILE" ]; then
  pass "dry-run creates .agent-status file"
else
  fail "dry-run creates .agent-status file (not found: $STATUS_FILE)"
fi

# ---------------------------------------------------------------------------
# Test 5: Status file has type=review
# ---------------------------------------------------------------------------
TYPE_VAL=$(python3 -c "import json,sys; d=json.load(open('$STATUS_FILE')); print(d.get('type','MISSING'))")
if [ "$TYPE_VAL" = "review" ]; then
  pass "status file has type=review"
else
  fail "status file has type=review (got: $TYPE_VAL)"
fi

# ---------------------------------------------------------------------------
# Test 6: Status file has correct target
# ---------------------------------------------------------------------------
TARGET_VAL=$(python3 -c "import json,sys; d=json.load(open('$STATUS_FILE')); print(d.get('target','MISSING'))")
if [ "$TARGET_VAL" = "$TEST_TARGET" ]; then
  pass "status file has correct target"
else
  fail "status file has correct target (got: $TARGET_VAL, expected: $TEST_TARGET)"
fi

# ---------------------------------------------------------------------------
# Test 7: Status file has tmux_session field
# ---------------------------------------------------------------------------
TMUX_VAL=$(python3 -c "import json,sys; d=json.load(open('$STATUS_FILE')); print(d.get('tmux_session','MISSING'))")
if [ "$TMUX_VAL" != "MISSING" ] && [ -n "$TMUX_VAL" ]; then
  pass "status file has tmux_session field"
else
  fail "status file has tmux_session field (got: $TMUX_VAL)"
fi

# ---------------------------------------------------------------------------
# Test 8: Dry-run creates .agent-prompt.txt
# ---------------------------------------------------------------------------
PROMPT_FILE="$TEST_REVIEW_DIR/.agent-prompt.txt"
if [ -f "$PROMPT_FILE" ]; then
  pass "dry-run creates .agent-prompt.txt"
else
  fail "dry-run creates .agent-prompt.txt (not found: $PROMPT_FILE)"
fi

# ---------------------------------------------------------------------------
# Test 9: Dry-run creates .agent-runner.sh
# ---------------------------------------------------------------------------
RUNNER_FILE="$TEST_REVIEW_DIR/.agent-runner.sh"
if [ -f "$RUNNER_FILE" ] && [ -x "$RUNNER_FILE" ]; then
  pass "dry-run creates .agent-runner.sh (executable)"
else
  fail "dry-run creates .agent-runner.sh (found=$([ -f "$RUNNER_FILE" ] && echo yes || echo no), exec=$([ -x "$RUNNER_FILE" ] && echo yes || echo no))"
fi

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
rm -rf "$TEST_REVIEW_DIR"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "Results: $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
