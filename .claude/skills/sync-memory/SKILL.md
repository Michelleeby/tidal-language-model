---
name: sync-memory
description: Detect and fix stale references in CLAUDE.md and MEMORY.md. Use when the user mentions "/sync-memory" or asks to update/validate CLAUDE.md.
---

# Sync Memory

When the user triggers this skill, follow these steps to detect and fix stale references in CLAUDE.md and MEMORY.md.

## 1. Run the Validator

Run the validation engine for a structured report:

```bash
python3 scripts/validate_memory.py --json
```

Parse the JSON output. If exit code 0 (no issues), report that everything is clean and stop.

## 2. Investigate Each Failure

For each failure in the report:

1. Read the CLAUDE.md line referenced in the failure
2. Read the actual source files (configs, Python files) to determine the **correct** current state
3. Determine the fix: update the prose to match reality

**Common fixes:**
- `architecture_constants`: A number in prose doesn't match config YAML. Read the config, update the prose.
- `file_references`: A backtick-quoted path doesn't exist. Find where the file moved, or remove the reference.
- `directory_references`: A directory (with trailing `/`) doesn't exist. Find where it moved, or remove the reference.
- `command_entrypoints`: A `python3 path/file.py` in a code block references a missing file. Find the correct path.
- `class_references`: A class name in backticks can't be found. Check if it was renamed or removed.
- `memory_line_count`: MEMORY.md is approaching the 200-line system limit. Consolidate or move details to topic files.

## 3. Update CLAUDE.md

Edit CLAUDE.md to fix all critical failures. Preserve the existing section structure (Commands, Architecture, Important Notes, MUST USE INSTRUCTIONS). Only change lines that are stale — do not rewrite sections unnecessarily.

## 4. Update MEMORY.md (if needed)

If the validator reports `memory_line_count` warnings, or if you find outdated information in MEMORY.md during investigation:

1. Remove or update outdated entries
2. If over 160 lines, move detailed notes to topic files under the memory directory and link from MEMORY.md

## 5. Re-validate

Run the validator again to confirm all fixes:

```bash
python3 scripts/validate_memory.py --json
```

Exit code should be 0. If not, repeat steps 2-4 for remaining issues.

## 6. Clean Up Marker

Delete the sync marker if it exists:

```bash
rm -f .claude/memory_sync_marker.json
```

## 7. Report Changes

Summarize what changed:
- Which lines of CLAUDE.md were updated
- What the old vs new values were
- Whether MEMORY.md was modified
- Final validation result (should be 0 critical, 0 warnings)
