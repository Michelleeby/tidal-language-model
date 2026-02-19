"""Tests for the plan archival hook (.claude/hooks/archive_plan.py)."""

import json
import os
import sys
import tempfile
import time
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

# Add the hook to the import path
HOOK_DIR = Path(__file__).resolve().parent.parent / ".claude" / "hooks"
sys.path.insert(0, str(HOOK_DIR))

import archive_plan


SAMPLE_PLAN = """\
# Plan: Single Modulation Gate Redesign

## Context

The current gating architecture uses three separate gate signals. This is overkill
for a small model trained on TinyStories.

## Design

Replace the 3-signal gate with a single modulation factor that controls all three
dimensions through learned linear projections.

## File Changes

| File | Action |
|------|--------|
| `plugins/tidal/GatingModulator.py` | Modify |
| `plugins/tidal/TransformerLM.py` | Modify |

## Verification

1. Run `python -m unittest plugins.tidal.tests.test_TransformerLM -v`
2. Check that gate signals are correctly projected.
"""

SAMPLE_PLAN_NO_TITLE = """\
Some plan without a proper H1 heading.

## Details

Just some details here.
"""

SAMPLE_PLAN_MANY_H2S = """\
# Plan: Big Refactor

## Context

Background info.

## Root Cause

The root cause is X.

## Solution

Do Y.

## Phase 1

Migrate data.

## Phase 2

Update API.

## Migration

Run the migration script.

## Notes

Some notes.

## Verification

Run the tests.

## Files Summary

List of files.
"""


class TestExtractTitle(unittest.TestCase):
    """extract_title should pull the title from an H1 line, stripping 'Plan:' prefix."""

    def test_extracts_title_from_h1(self):
        title = archive_plan.extract_title(SAMPLE_PLAN)
        self.assertEqual(title, "Single Modulation Gate Redesign")

    def test_strips_plan_prefix(self):
        md = "# Plan: My Great Redesign\n\nBody text."
        title = archive_plan.extract_title(md)
        self.assertEqual(title, "My Great Redesign")

    def test_fallback_when_no_h1(self):
        title = archive_plan.extract_title(SAMPLE_PLAN_NO_TITLE)
        # Should fall back to first non-empty line, truncated
        self.assertIn("Some plan", title)

    def test_strips_whitespace(self):
        md = "#   Plan:   Spaced Out Title  \n\nBody."
        title = archive_plan.extract_title(md)
        self.assertEqual(title, "Spaced Out Title")


class TestExtractContentTags(unittest.TestCase):
    """extract_content_tags should return slugified H2 headings, filtering generic ones."""

    def test_extracts_meaningful_h2s(self):
        tags = archive_plan.extract_content_tags(SAMPLE_PLAN)
        self.assertIn("design", tags)
        self.assertIn("file_changes", tags)

    def test_filters_generic_headings(self):
        tags = archive_plan.extract_content_tags(SAMPLE_PLAN)
        self.assertNotIn("context", tags)
        self.assertNotIn("verification", tags)

    def test_max_four_tags(self):
        tags = archive_plan.extract_content_tags(SAMPLE_PLAN_MANY_H2S)
        self.assertLessEqual(len(tags), 4)

    def test_filters_notes_and_files_summary(self):
        tags = archive_plan.extract_content_tags(SAMPLE_PLAN_MANY_H2S)
        self.assertNotIn("notes", tags)
        self.assertNotIn("files_summary", tags)

    def test_keeps_root_cause_and_solution(self):
        tags = archive_plan.extract_content_tags(SAMPLE_PLAN_MANY_H2S)
        self.assertIn("root_cause", tags)
        self.assertIn("solution", tags)

    def test_returns_empty_for_no_h2s(self):
        tags = archive_plan.extract_content_tags("# Title\n\nJust a paragraph.")
        self.assertEqual(tags, [])


class TestSluggify(unittest.TestCase):
    """slugify should produce filesystem-safe slugs."""

    def test_basic_slugify(self):
        self.assertEqual(archive_plan.slugify("Hello World"), "hello_world")

    def test_strips_special_chars(self):
        self.assertEqual(archive_plan.slugify("Plan: Test (v2)"), "plan_test_v2")

    def test_max_length(self):
        long = "a" * 100
        result = archive_plan.slugify(long, max_len=20)
        self.assertEqual(len(result), 20)


class TestFindLatestPlan(unittest.TestCase):
    """find_latest_plan should find the most recently modified .md in the plans directory."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plans_dir = Path(self.tmp.name) / "plans"
        self.plans_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_most_recent_plan(self):
        old = self.plans_dir / "old-plan.md"
        old.write_text("# Old Plan\n")

        # Ensure different mtimes
        time.sleep(0.05)

        new = self.plans_dir / "new-plan.md"
        new.write_text("# New Plan\n")

        result = archive_plan.find_latest_plan(self.plans_dir)
        self.assertEqual(result, new)

    def test_returns_none_for_empty_dir(self):
        result = archive_plan.find_latest_plan(self.plans_dir)
        self.assertIsNone(result)

    def test_returns_none_for_nonexistent_dir(self):
        result = archive_plan.find_latest_plan(Path(self.tmp.name) / "nonexistent")
        self.assertIsNone(result)

    def test_ignores_non_md_files(self):
        txt = self.plans_dir / "notes.txt"
        txt.write_text("not a plan")
        result = archive_plan.find_latest_plan(self.plans_dir)
        self.assertIsNone(result)

    def test_skips_already_archived_plans(self):
        plan = self.plans_dir / "done.md"
        plan.write_text("# Done\n")
        marker = self.plans_dir / "done.plan_archived"
        marker.touch()

        result = archive_plan.find_latest_plan(self.plans_dir)
        self.assertIsNone(result)


class TestStalenessCheck(unittest.TestCase):
    """is_stale should return True if the plan file is older than the threshold."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan_file = Path(self.tmp.name) / "plan.md"
        self.plan_file.write_text("# Plan\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_fresh_plan_is_not_stale(self):
        self.assertFalse(archive_plan.is_stale(self.plan_file, max_age_seconds=120))

    def test_old_plan_is_stale(self):
        # Set mtime to 200 seconds ago
        old_time = time.time() - 200
        os.utime(self.plan_file, (old_time, old_time))
        self.assertTrue(archive_plan.is_stale(self.plan_file, max_age_seconds=120))

    def test_boundary_not_stale(self):
        # Set mtime to exactly 119 seconds ago
        boundary_time = time.time() - 119
        os.utime(self.plan_file, (boundary_time, boundary_time))
        self.assertFalse(archive_plan.is_stale(self.plan_file, max_age_seconds=120))


class TestArchivePlan(unittest.TestCase):
    """archive_plan should copy the plan file with a timestamped name and create a marker."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.project_dir = self.tmp.name
        self.plans_src = Path(self.tmp.name) / "claude_plans"
        self.plans_src.mkdir()

        # Write a sample plan
        self.plan_file = self.plans_src / "fuzzy-jumping-cat.md"
        self.plan_file.write_text(SAMPLE_PLAN)

    def tearDown(self):
        self.tmp.cleanup()

    def test_copies_plan_to_research_dir(self):
        archive_plan.archive_plan(
            project_dir=self.project_dir,
            plan_path=self.plan_file,
        )
        output_dir = Path(self.project_dir) / "research" / "plans"
        md_files = list(output_dir.glob("*.md"))
        self.assertEqual(len(md_files), 1)

    def test_filename_contains_title_slug(self):
        archive_plan.archive_plan(
            project_dir=self.project_dir,
            plan_path=self.plan_file,
        )
        output_dir = Path(self.project_dir) / "research" / "plans"
        md_files = list(output_dir.glob("*.md"))
        self.assertIn("single_modulation_gate_redesign", md_files[0].name)

    def test_filename_contains_content_tags(self):
        archive_plan.archive_plan(
            project_dir=self.project_dir,
            plan_path=self.plan_file,
        )
        output_dir = Path(self.project_dir) / "research" / "plans"
        md_files = list(output_dir.glob("*.md"))
        self.assertIn("__design", md_files[0].name)

    def test_creates_marker_file(self):
        archive_plan.archive_plan(
            project_dir=self.project_dir,
            plan_path=self.plan_file,
        )
        marker = self.plan_file.with_suffix(".plan_archived")
        self.assertTrue(marker.exists())

    def test_no_double_archival(self):
        archive_plan.archive_plan(
            project_dir=self.project_dir,
            plan_path=self.plan_file,
        )
        archive_plan.archive_plan(
            project_dir=self.project_dir,
            plan_path=self.plan_file,
        )
        output_dir = Path(self.project_dir) / "research" / "plans"
        md_files = list(output_dir.glob("*.md"))
        self.assertEqual(len(md_files), 1)

    def test_skips_stale_plan(self):
        old_time = time.time() - 200
        os.utime(self.plan_file, (old_time, old_time))

        archive_plan.archive_plan(
            project_dir=self.project_dir,
            plan_path=self.plan_file,
        )
        output_dir = Path(self.project_dir) / "research" / "plans"
        self.assertFalse(output_dir.exists())

    def test_archived_content_matches_source(self):
        archive_plan.archive_plan(
            project_dir=self.project_dir,
            plan_path=self.plan_file,
        )
        output_dir = Path(self.project_dir) / "research" / "plans"
        md_files = list(output_dir.glob("*.md"))
        archived_content = md_files[0].read_text()
        self.assertEqual(archived_content, SAMPLE_PLAN)

    def test_creates_research_plans_directory(self):
        """Should create research/plans/ if it doesn't exist."""
        output_dir = Path(self.project_dir) / "research" / "plans"
        self.assertFalse(output_dir.exists())

        archive_plan.archive_plan(
            project_dir=self.project_dir,
            plan_path=self.plan_file,
        )
        self.assertTrue(output_dir.exists())


class TestMainStdinParsing(unittest.TestCase):
    """main() should read hook input from stdin and locate the latest plan."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.project_dir = self.tmp.name

        # Create a plan in a mock plans dir
        self.plans_dir = Path(self.tmp.name) / "mock_plans"
        self.plans_dir.mkdir()
        self.plan_file = self.plans_dir / "test-plan.md"
        self.plan_file.write_text(SAMPLE_PLAN)

    def tearDown(self):
        self.tmp.cleanup()

    @patch("archive_plan.archive_plan")
    @patch("archive_plan.find_latest_plan")
    def test_main_calls_archive_with_latest_plan(self, mock_find, mock_archive):
        mock_find.return_value = self.plan_file

        hook_input = {"cwd": self.project_dir}
        with patch("sys.stdin", StringIO(json.dumps(hook_input))):
            with self.assertRaises(SystemExit) as ctx:
                archive_plan.main()
            self.assertEqual(ctx.exception.code, 0)

        mock_archive.assert_called_once_with(
            project_dir=self.project_dir,
            plan_path=self.plan_file,
        )

    @patch("archive_plan.archive_plan")
    @patch("archive_plan.find_latest_plan")
    def test_main_skips_when_no_plan_found(self, mock_find, mock_archive):
        mock_find.return_value = None

        hook_input = {"cwd": self.project_dir}
        with patch("sys.stdin", StringIO(json.dumps(hook_input))):
            with self.assertRaises(SystemExit) as ctx:
                archive_plan.main()
            self.assertEqual(ctx.exception.code, 0)

        mock_archive.assert_not_called()

    @patch("archive_plan.archive_plan")
    @patch("archive_plan.find_latest_plan")
    @patch("archive_plan._debug_log")
    def test_main_uses_cwd_from_stdin(self, mock_log, mock_find, mock_archive):
        mock_find.return_value = None

        hook_input = {"cwd": self.project_dir}
        with patch("sys.stdin", StringIO(json.dumps(hook_input))):
            with self.assertRaises(SystemExit) as ctx:
                archive_plan.main()
            self.assertEqual(ctx.exception.code, 0)

        # _debug_log should receive the cwd from stdin
        mock_log.assert_any_call(self.project_dir, unittest.mock.ANY)
        # find_latest_plan should be called (with the default plans dir)
        mock_find.assert_called_once()

    @patch("archive_plan.archive_plan")
    @patch("archive_plan.find_latest_plan")
    def test_main_handles_malformed_stdin(self, mock_find, mock_archive):
        mock_find.return_value = None

        with patch("sys.stdin", StringIO("not json")):
            with self.assertRaises(SystemExit) as ctx:
                archive_plan.main()
            self.assertEqual(ctx.exception.code, 0)

        # Should still attempt to run (with fallback cwd)
        mock_find.assert_called_once()


class TestDebugLog(unittest.TestCase):
    """archive_plan should write debug log entries."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.project_dir = self.tmp.name
        self.log_file = (
            Path(self.project_dir) / ".claude" / "hooks" / "archive_plan_debug.log"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_debug_log_created(self):
        archive_plan._debug_log(self.project_dir, "test message")
        self.assertTrue(self.log_file.exists())
        content = self.log_file.read_text()
        self.assertIn("test message", content)


if __name__ == "__main__":
    unittest.main()
