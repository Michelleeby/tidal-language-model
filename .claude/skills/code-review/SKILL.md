---
name: code-review
description: Review a PR or branch using multi-model AI analysis (bugs, hypothesis alignment, ADR compliance) via the plan-review CLI. Runs in the background via tmux. Use when the user mentions "/code-review", asks to review a PR, or wants automated code review.
---

# Code Review Skill

Launch a background multi-model code review for a pull request or branch. Uses the plan-review CLI directly (GPT-4o + Gemini Flash, no Claude agent orchestrator).

## Steps

### 1. Determine the Target

Identify what to review from the user's message:
- A PR number: e.g., `/code-review 42` or "review PR 42"
- A branch name: e.g., `/code-review impl/my-feature`
- Current branch vs main: if no target specified, use the current branch

If no target is given, run `git branch --show-current` to get the current branch name.

### 2. Validate the Target

- If target is a PR number: run `gh pr view <N>` to confirm it exists. If not found, tell the user.
- If target is a branch name: run `git rev-parse --verify <branch>` to confirm it exists. If not found, suggest they check the branch name.

### 3. Launch the Review

Run:
```
bash local/scripts/code_review_agent.sh "<target>"
```

This will:
- Create `.claude/reviews/<target>/` with runner files
- Launch the plan-review CLI in a tmux session named `review-<target>`
- The CLI calls OpenAI (GPT-4o) and Google (Gemini Flash) directly
- Output will be written to `.claude/reviews/<target>/.review-output.md`
- If target is a PR number, the report will be posted as a PR comment

### 4. Report to User

Tell the user:
- The tmux session name: `review-<target>`
- The log file path: `.claude/reviews/<target>/.agent-output.log`
- The output path: `.claude/reviews/<target>/.review-output.md`
- They can check progress with: `tmux attach -t review-<target>` or `tail -f .claude/reviews/<target>/.agent-output.log`

Example message:
```
Code review launched for <target>.
- Session:  tmux attach -t review-<target>
- Log:      tail -f .claude/reviews/<target>/.agent-output.log
- Output:   .claude/reviews/<target>/.review-output.md

The CLI reviews for bugs, hypothesis alignment, and ADR compliance using GPT-4o and Gemini Flash.
```

## Notes

- Reviews are read-only — they never modify source files
- Budget defaults to "standard" (2 models per dimension)
- ADR context is automatically injected
- For PR targets, the report is posted as a comment automatically
- No Anthropic API tokens are consumed — only OpenAI and Google
