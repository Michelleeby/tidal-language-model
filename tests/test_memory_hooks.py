"""Tests for memory sync hook scripts.

Tests:
  - stop_memory_check.py: exit codes, fast-mode behavior, timeout safety
  - memory_marker.py: marker creation, cleanup, chaining
  - session_start_memory.py: reminder output, staleness expiry
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

HOOKS_DIR = Path(__file__).resolve().parent.parent / ".claude" / "hooks"
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


class _TempProjectMixin:
    """Shared helper to create a temporary project with controlled files."""

    def make_project(
        self,
        claude_md="",
        base_config_yaml="",
        rl_config_yaml="",
        extra_files=None,
        extra_dirs=None,
        marker=None,
    ):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmpdir, ignore_errors=True))
        root = Path(tmpdir)

        if claude_md:
            (root / "CLAUDE.md").write_text(textwrap.dedent(claude_md))

        config_dir = root / "plugins" / "tidal" / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        if base_config_yaml:
            (config_dir / "base_config.yaml").write_text(
                textwrap.dedent(base_config_yaml)
            )
        if rl_config_yaml:
            (config_dir / "rl_config.yaml").write_text(
                textwrap.dedent(rl_config_yaml)
            )

        for rel_path, content in (extra_files or {}).items():
            fpath = root / rel_path
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)

        for dirname in extra_dirs or []:
            (root / dirname).mkdir(parents=True, exist_ok=True)

        # .claude directory for marker
        (root / ".claude").mkdir(parents=True, exist_ok=True)

        # Copy validate_memory.py into the temp project so hooks can find it
        real_script = SCRIPTS_DIR / "validate_memory.py"
        if real_script.exists():
            scripts_dir = root / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(str(real_script), str(scripts_dir / "validate_memory.py"))

        if marker is not None:
            marker_path = root / ".claude" / "memory_sync_marker.json"
            marker_path.write_text(json.dumps(marker))

        return root


# ---------------------------------------------------------------------------
# Stop hook tests
# ---------------------------------------------------------------------------


class TestStopMemoryCheck(_TempProjectMixin, unittest.TestCase):
    """Tests for .claude/hooks/stop_memory_check.py."""

    def _run_hook(self, project_root, stdin_data=None):
        """Run the stop hook as a subprocess, return (exit_code, stdout, stderr)."""
        hook = HOOKS_DIR / "stop_memory_check.py"
        env = os.environ.copy()
        env["CLAUDE_PROJECT_DIR"] = str(project_root)
        if stdin_data is None:
            stdin_data = json.dumps({"cwd": str(project_root)})
        result = subprocess.run(
            [sys.executable, str(hook)],
            input=stdin_data,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        return result.returncode, result.stdout, result.stderr

    def test_clean_project_exits_0(self):
        """No staleness -> exit 0."""
        root = self.make_project(
            claude_md="""
            # Clean
            See `plugins/tidal/Main.py` for details.
            """,
            extra_files={"plugins/tidal/Main.py": "# main"},
        )
        code, stdout, stderr = self._run_hook(root)
        self.assertEqual(code, 0)

    def test_stale_reference_exits_2(self):
        """Critical staleness -> exit 2 to block."""
        root = self.make_project(
            claude_md="""
            Legacy code is in `legacy_research/` — do not use.
            """,
        )
        code, stdout, stderr = self._run_hook(root)
        self.assertEqual(code, 2)
        # stdout should contain a message for Claude
        self.assertIn("stale", stdout.lower())

    def test_graceful_on_invalid_stdin(self):
        """Bad stdin -> exit 0 (graceful fallback)."""
        root = self.make_project(claude_md="# OK\n")
        code, stdout, stderr = self._run_hook(root, stdin_data="not json")
        self.assertEqual(code, 0)


# ---------------------------------------------------------------------------
# Memory marker tests
# ---------------------------------------------------------------------------


class TestMemoryMarker(_TempProjectMixin, unittest.TestCase):
    """Tests for .claude/hooks/memory_marker.py."""

    def _run_hook(self, project_root, stdin_data=None):
        hook = HOOKS_DIR / "memory_marker.py"
        env = os.environ.copy()
        env["CLAUDE_PROJECT_DIR"] = str(project_root)
        if stdin_data is None:
            stdin_data = json.dumps({"cwd": str(project_root)})
        result = subprocess.run(
            [sys.executable, str(hook)],
            input=stdin_data,
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
        return result.returncode, result.stdout, result.stderr

    def test_creates_marker_on_staleness(self):
        """When staleness found, writes marker file."""
        root = self.make_project(
            claude_md="""
            Legacy code is in `legacy_research/` — do not use.
            """,
        )
        code, _, _ = self._run_hook(root)
        self.assertEqual(code, 0)  # Marker hook always exits 0
        marker = root / ".claude" / "memory_sync_marker.json"
        self.assertTrue(marker.exists(), "Marker file should be created")
        data = json.loads(marker.read_text())
        self.assertIn("results", data)
        self.assertIn("timestamp", data)

    def test_deletes_marker_when_clean(self):
        """When clean, deletes existing marker."""
        root = self.make_project(
            claude_md="# Clean doc\n",
            marker={"results": [], "timestamp": "old"},
        )
        marker = root / ".claude" / "memory_sync_marker.json"
        self.assertTrue(marker.exists())  # Pre-condition
        code, _, _ = self._run_hook(root)
        self.assertEqual(code, 0)
        self.assertFalse(marker.exists(), "Marker should be deleted when clean")

    def test_always_exits_0(self):
        """Marker hook should never block Claude Code."""
        root = self.make_project(
            claude_md="""
            `legacy_research/` and `missing/dir/` are both gone.
            """,
        )
        code, _, _ = self._run_hook(root)
        self.assertEqual(code, 0)


# ---------------------------------------------------------------------------
# Session start reminder tests
# ---------------------------------------------------------------------------


class TestSessionStartMemory(_TempProjectMixin, unittest.TestCase):
    """Tests for .claude/hooks/session_start_memory.py."""

    def _run_hook(self, project_root, stdin_data=None):
        hook = HOOKS_DIR / "session_start_memory.py"
        env = os.environ.copy()
        env["CLAUDE_PROJECT_DIR"] = str(project_root)
        if stdin_data is None:
            stdin_data = json.dumps({"cwd": str(project_root)})
        result = subprocess.run(
            [sys.executable, str(hook)],
            input=stdin_data,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        return result.returncode, result.stdout, result.stderr

    def test_prints_reminder_when_marker_exists(self):
        """Marker present -> stdout contains reminder for Claude."""
        root = self.make_project(
            claude_md="# OK\n",
            marker={
                "results": [
                    {"check": "directory_references", "severity": "critical",
                     "message": "Directory `legacy_research/` not found", "line": 79}
                ],
                "timestamp": "2026-02-20T12:00:00Z",
            },
        )
        code, stdout, stderr = self._run_hook(root)
        self.assertEqual(code, 0)
        self.assertIn("CLAUDE.md", stdout)
        self.assertIn("legacy_research/", stdout)

    def test_no_output_without_marker(self):
        """No marker -> no stdout output."""
        root = self.make_project(claude_md="# OK\n")
        code, stdout, stderr = self._run_hook(root)
        self.assertEqual(code, 0)
        self.assertEqual(stdout.strip(), "")

    def test_ignores_expired_marker(self):
        """Marker older than 7 days -> ignored."""
        root = self.make_project(
            claude_md="# OK\n",
            marker={
                "results": [{"check": "test", "severity": "critical",
                             "message": "old", "line": 1}],
                "timestamp": "2026-01-01T00:00:00Z",  # >7 days old
            },
        )
        code, stdout, stderr = self._run_hook(root)
        self.assertEqual(code, 0)
        self.assertEqual(stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
