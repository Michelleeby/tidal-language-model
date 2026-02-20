"""Tests for scripts/validate_memory.py — Memory sync validation engine.

Uses tempdir fixtures to create controlled CLAUDE.md / MEMORY.md / config files
and verifies each check independently.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

# We'll import the module under test once it exists
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))


class _TempProjectMixin:
    """Mixin that creates a temporary project directory with configurable files."""

    def make_project(
        self,
        claude_md="",
        memory_md="",
        base_config_yaml="",
        rl_config_yaml="",
        extra_files=None,
        extra_dirs=None,
    ):
        """Create a temporary project directory and return its Path."""
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmpdir, ignore_errors=True))

        root = Path(tmpdir)

        # CLAUDE.md at root
        if claude_md:
            (root / "CLAUDE.md").write_text(textwrap.dedent(claude_md))

        # MEMORY.md in the expected location
        memory_dir = root / ".claude" / "projects" / "test" / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        if memory_md:
            (memory_dir / "MEMORY.md").write_text(textwrap.dedent(memory_md))

        # Config files
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

        # Extra files (dict of relative_path -> content)
        for rel_path, content in (extra_files or {}).items():
            fpath = root / rel_path
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)

        # Extra directories
        for dirname in extra_dirs or []:
            (root / dirname).mkdir(parents=True, exist_ok=True)

        return root


# ---------------------------------------------------------------------------
# Tests for individual check classes
# ---------------------------------------------------------------------------


class TestFileReferencesCheck(_TempProjectMixin, unittest.TestCase):
    """Check that backtick-quoted file paths in CLAUDE.md are validated."""

    def test_existing_file_passes(self):
        from validate_memory import FileReferencesCheck

        root = self.make_project(
            claude_md="""
            ## Architecture
            See `plugins/tidal/Main.py` for the entry point.
            """,
            extra_files={"plugins/tidal/Main.py": "# main"},
        )
        check = FileReferencesCheck(root)
        results = check.run()
        self.assertEqual(len(results), 0, f"Expected no failures, got: {results}")

    def test_missing_file_fails(self):
        from validate_memory import FileReferencesCheck

        root = self.make_project(
            claude_md="""
            ## Architecture
            See `plugins/tidal/NonExistent.py` for details.
            """,
        )
        check = FileReferencesCheck(root)
        results = check.run()
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["severity"], "critical")
        self.assertIn("NonExistent.py", results[0]["message"])

    def test_ignores_non_path_backticks(self):
        from validate_memory import FileReferencesCheck

        root = self.make_project(
            claude_md="""
            Use `torch.compile` for speed.
            The vocab is `50257` tokens.
            """,
        )
        check = FileReferencesCheck(root)
        results = check.run()
        self.assertEqual(len(results), 0)

    def test_fast_flag(self):
        from validate_memory import FileReferencesCheck

        check = FileReferencesCheck(Path("/tmp"))
        self.assertTrue(check.fast)


class TestCommandEntrypointsCheck(_TempProjectMixin, unittest.TestCase):
    """Check that python3 entrypoints in code blocks resolve to real files."""

    def test_valid_entrypoint_passes(self):
        from validate_memory import CommandEntrypointsCheck

        root = self.make_project(
            claude_md="""
            ```bash
            python3 plugins/tidal/Main.py --config base.yaml
            ```
            """,
            extra_files={"plugins/tidal/Main.py": "# main"},
        )
        check = CommandEntrypointsCheck(root)
        results = check.run()
        self.assertEqual(len(results), 0)

    def test_missing_entrypoint_fails(self):
        from validate_memory import CommandEntrypointsCheck

        root = self.make_project(
            claude_md="""
            ```bash
            python3 plugins/tidal/Missing.py --config foo.yaml
            ```
            """,
        )
        check = CommandEntrypointsCheck(root)
        results = check.run()
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["severity"], "critical")

    def test_fast_flag(self):
        from validate_memory import CommandEntrypointsCheck

        check = CommandEntrypointsCheck(Path("/tmp"))
        self.assertTrue(check.fast)


class TestDirectoryReferencesCheck(_TempProjectMixin, unittest.TestCase):
    """Check directory references in prose (ending with /)."""

    def test_existing_directory_passes(self):
        from validate_memory import DirectoryReferencesCheck

        root = self.make_project(
            claude_md="""
            Model code lives in `plugins/tidal/` with a manifest.
            """,
            extra_dirs=["plugins/tidal"],
        )
        check = DirectoryReferencesCheck(root)
        results = check.run()
        self.assertEqual(len(results), 0)

    def test_missing_directory_fails(self):
        from validate_memory import DirectoryReferencesCheck

        root = self.make_project(
            claude_md="""
            Legacy code is in `legacy_research/` — do not use.
            """,
        )
        check = DirectoryReferencesCheck(root)
        results = check.run()
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["severity"], "critical")
        self.assertIn("legacy_research/", results[0]["message"])

    def test_fast_flag(self):
        from validate_memory import DirectoryReferencesCheck

        check = DirectoryReferencesCheck(Path("/tmp"))
        self.assertTrue(check.fast)


class TestArchitectureConstantsCheck(_TempProjectMixin, unittest.TestCase):
    """Check that numeric constants in prose match config YAML values."""

    def test_matching_gate_dim_passes(self):
        from validate_memory import ArchitectureConstantsCheck

        root = self.make_project(
            claude_md="""
            The agent outputs 1 gate signal for modulation.
            The model has 6 `GatedTransformerBlock` layers.
            The MLP converts the 1D gate signal into scaling factors.
            """,
            base_config_yaml="""
            GATE_DIM: 1
            NUM_TRANSFORMER_BLOCKS: 6
            VOCAB_SIZE: 50257
            """,
            rl_config_yaml="""
            RL_ACTION_DIM: 1
            """,
        )
        check = ArchitectureConstantsCheck(root)
        results = check.run()
        # Should have no gate-dim or layer-count mismatches
        gate_failures = [r for r in results if "gate" in r["message"].lower() or "GATE_DIM" in r["message"]]
        self.assertEqual(len(gate_failures), 0, f"Unexpected gate failures: {gate_failures}")

    def test_mismatched_gate_dim_fails(self):
        from validate_memory import ArchitectureConstantsCheck

        root = self.make_project(
            claude_md="""
            The agent controls 3 gate signals — [creativity, focus, stability].
            These are small MLPs (3→32→embed_dim→sigmoid).
            """,
            base_config_yaml="""
            GATE_DIM: 1
            """,
            rl_config_yaml="""
            RL_ACTION_DIM: 1
            """,
        )
        check = ArchitectureConstantsCheck(root)
        results = check.run()
        self.assertTrue(len(results) >= 1, "Expected gate-dim mismatch failure")
        self.assertEqual(results[0]["severity"], "critical")

    def test_fast_flag(self):
        from validate_memory import ArchitectureConstantsCheck

        check = ArchitectureConstantsCheck(Path("/tmp"))
        self.assertTrue(check.fast)


class TestClassReferencesCheck(_TempProjectMixin, unittest.TestCase):
    """Check that class names in backticks can be found in Python files."""

    def test_existing_class_passes(self):
        from validate_memory import ClassReferencesCheck

        root = self.make_project(
            claude_md="""
            The `TransformerLM` model is the core component.
            """,
            extra_files={
                "plugins/tidal/TransformerLM.py": "class TransformerLM:\n    pass\n"
            },
        )
        check = ClassReferencesCheck(root)
        results = check.run()
        self.assertEqual(len(results), 0)

    def test_missing_class_warns(self):
        from validate_memory import ClassReferencesCheck

        root = self.make_project(
            claude_md="""
            The `GhostModule` is used for inference.
            """,
        )
        check = ClassReferencesCheck(root)
        results = check.run()
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["severity"], "warning")

    def test_not_fast(self):
        from validate_memory import ClassReferencesCheck

        check = ClassReferencesCheck(Path("/tmp"))
        self.assertFalse(check.fast)


class TestMemoryLineCountCheck(_TempProjectMixin, unittest.TestCase):
    """Check MEMORY.md line count against threshold."""

    def test_short_memory_passes(self):
        from validate_memory import MemoryLineCountCheck

        root = self.make_project(memory_md="# Memory\n\nShort file.\n")
        check = MemoryLineCountCheck(root)
        results = check.run()
        self.assertEqual(len(results), 0)

    def test_long_memory_warns(self):
        from validate_memory import MemoryLineCountCheck

        long_content = "\n".join([f"Line {i}" for i in range(170)])
        root = self.make_project(memory_md=long_content)
        check = MemoryLineCountCheck(root)
        results = check.run()
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["severity"], "warning")
        self.assertIn("170", results[0]["message"])

    def test_fast_flag(self):
        from validate_memory import MemoryLineCountCheck

        check = MemoryLineCountCheck(Path("/tmp"))
        self.assertTrue(check.fast)


# ---------------------------------------------------------------------------
# Tests for the validator engine (run_all, fast mode, JSON output)
# ---------------------------------------------------------------------------


class TestValidatorEngine(_TempProjectMixin, unittest.TestCase):
    """Test the top-level validator that orchestrates all checks."""

    def test_run_all_collects_results(self):
        from validate_memory import MemoryValidator

        root = self.make_project(
            claude_md="""
            See `plugins/tidal/Main.py` for details.
            Legacy code is in `legacy_research/` — do not use.
            """,
            extra_files={"plugins/tidal/Main.py": "# main"},
        )
        validator = MemoryValidator(root)
        report = validator.run_all()
        # Should find the missing legacy_research/ directory
        critical = [r for r in report["results"] if r["severity"] == "critical"]
        self.assertTrue(len(critical) >= 1)

    def test_fast_mode_skips_slow_checks(self):
        from validate_memory import MemoryValidator

        root = self.make_project(
            claude_md="""
            The `GhostModule` is used for inference.
            """,
        )
        validator = MemoryValidator(root)
        report_full = validator.run_all(fast=False)
        report_fast = validator.run_all(fast=True)
        # ClassReferencesCheck is not fast, so fast mode should have fewer results
        full_checks = set(r["check"] for r in report_full["results"])
        fast_checks = set(r["check"] for r in report_fast["results"])
        # class_references should be in full but not fast
        if "class_references" in full_checks:
            self.assertNotIn("class_references", fast_checks)

    def test_exit_code_0_when_clean(self):
        from validate_memory import compute_exit_code

        report = {"results": [], "summary": {"critical": 0, "warning": 0}}
        self.assertEqual(compute_exit_code(report), 0)

    def test_exit_code_1_on_critical(self):
        from validate_memory import compute_exit_code

        report = {
            "results": [{"severity": "critical"}],
            "summary": {"critical": 1, "warning": 0},
        }
        self.assertEqual(compute_exit_code(report), 1)

    def test_exit_code_2_on_warnings_only(self):
        from validate_memory import compute_exit_code

        report = {
            "results": [{"severity": "warning"}],
            "summary": {"critical": 0, "warning": 1},
        }
        self.assertEqual(compute_exit_code(report), 2)

    def test_json_output_structure(self):
        from validate_memory import MemoryValidator

        root = self.make_project(claude_md="# Clean\n")
        validator = MemoryValidator(root)
        report = validator.run_all()
        # Must have these top-level keys
        self.assertIn("results", report)
        self.assertIn("summary", report)
        self.assertIn("critical", report["summary"])
        self.assertIn("warning", report["summary"])


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------


class TestCLI(unittest.TestCase):
    """Test the script as a subprocess."""

    def test_cli_json_flag(self):
        script = SCRIPT_DIR / "validate_memory.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "CLAUDE.md").write_text("# Clean doc\n")
            result = subprocess.run(
                [sys.executable, str(script), "--json", "--project-dir", str(root)],
                capture_output=True,
                text=True,
            )
            # Should produce valid JSON on stdout
            output = json.loads(result.stdout)
            self.assertIn("results", output)

    def test_cli_fast_flag(self):
        script = SCRIPT_DIR / "validate_memory.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "CLAUDE.md").write_text("# Clean doc\n")
            result = subprocess.run(
                [sys.executable, str(script), "--fast", "--json", "--project-dir", str(root)],
                capture_output=True,
                text=True,
            )
            output = json.loads(result.stdout)
            self.assertIn("results", output)


if __name__ == "__main__":
    unittest.main()
