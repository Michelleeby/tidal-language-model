#!/usr/bin/env python3
"""
Memory sync validation engine.

Checks CLAUDE.md and MEMORY.md references against the real filesystem and
config files. Strategy pattern: each check is an independent class.

Usage:
    python3 scripts/validate_memory.py [--fast] [--json] [--project-dir DIR]

Exit codes: 0 = clean, 1 = critical failures, 2 = warnings only.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Base check interface (Strategy pattern)
# ---------------------------------------------------------------------------


class BaseCheck(ABC):
    """Each check has an id, severity, fast flag, and run() method."""

    id: str = ""
    severity: str = "critical"  # "critical" or "warning"
    fast: bool = True  # Include in --fast mode?

    def __init__(self, project_root: Path):
        self.project_root = project_root

    @abstractmethod
    def run(self) -> list[dict[str, Any]]:
        """Return a list of failure dicts, or [] if clean."""
        ...

    def _read_claude_md(self) -> str:
        path = self.project_root / "CLAUDE.md"
        if path.exists():
            return path.read_text()
        return ""

    def _read_claude_md_lines(self) -> list[str]:
        path = self.project_root / "CLAUDE.md"
        if path.exists():
            return path.read_text().splitlines()
        return []

    def _failure(self, message: str, line: int | None = None) -> dict[str, Any]:
        return {
            "check": self.id,
            "severity": self.severity,
            "message": message,
            "line": line,
        }


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------


class FileReferencesCheck(BaseCheck):
    """Backtick-quoted file paths in CLAUDE.md -> os.path.exists()."""

    id = "file_references"
    severity = "critical"
    fast = True

    # Match backtick-quoted paths that look like file references:
    # - contain a / (path separator)
    # - have a file extension
    # - exclude things that look like python dotted imports or plain words
    _PATH_RE = re.compile(r"`([^`]+\.[a-zA-Z]{1,5})`")

    # Patterns that should NOT be treated as file paths
    _IGNORE_PATTERNS = [
        re.compile(r"^\d"),  # starts with digit
        re.compile(r"^[a-z_]+\.[a-z_]+\("),  # method call like torch.compile(
        re.compile(r"^[a-z_]+\.[a-z_]+$"),  # dotted name without /
        re.compile(r"^<"),  # placeholder like <model_checkpoint>
        re.compile(r"roneneldan/"),  # HuggingFace dataset
    ]

    def _looks_like_path(self, text: str) -> bool:
        """Heuristic: does this backtick content look like a filesystem path?"""
        # Must contain a / to be a path
        if "/" not in text:
            return False
        # Filter out known non-paths
        for pat in self._IGNORE_PATTERNS:
            if pat.search(text):
                return False
        # Must have an extension-like suffix
        return True

    def run(self) -> list[dict[str, Any]]:
        failures = []
        lines = self._read_claude_md_lines()
        for i, line in enumerate(lines, 1):
            for match in self._PATH_RE.finditer(line):
                ref = match.group(1)
                if not self._looks_like_path(ref):
                    continue
                # Strip trailing text after the path (e.g. "Main.py →" )
                ref_clean = ref.split("→")[0].split(" ")[0].rstrip(")")
                full = self.project_root / ref_clean
                if not full.exists():
                    failures.append(
                        self._failure(
                            f"File reference `{ref_clean}` not found", line=i
                        )
                    )
        return failures


class CommandEntrypointsCheck(BaseCheck):
    """python3 entrypoints in code blocks -> file exists?"""

    id = "command_entrypoints"
    severity = "critical"
    fast = True

    _CMD_RE = re.compile(r"python3?\s+([\w/.-]+\.py)")

    def run(self) -> list[dict[str, Any]]:
        failures = []
        lines = self._read_claude_md_lines()
        in_code_block = False
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if not in_code_block:
                continue
            for match in self._CMD_RE.finditer(line):
                script = match.group(1)
                full = self.project_root / script
                if not full.exists():
                    failures.append(
                        self._failure(
                            f"Command entrypoint `{script}` not found", line=i
                        )
                    )
        return failures


class DirectoryReferencesCheck(BaseCheck):
    """Directories referenced with trailing / in backticks -> exists?"""

    id = "directory_references"
    severity = "critical"
    fast = True

    _DIR_RE = re.compile(r"`([\w./-]+/)`")

    def run(self) -> list[dict[str, Any]]:
        failures = []
        lines = self._read_claude_md_lines()
        for i, line in enumerate(lines, 1):
            for match in self._DIR_RE.finditer(line):
                dirname = match.group(1)
                full = self.project_root / dirname
                if not full.exists() or not full.is_dir():
                    failures.append(
                        self._failure(
                            f"Directory `{dirname}` not found", line=i
                        )
                    )
        return failures


class ArchitectureConstantsCheck(BaseCheck):
    """Numbers in prose vs config YAML (GATE_DIM, NUM_TRANSFORMER_BLOCKS, etc.)."""

    id = "architecture_constants"
    severity = "critical"
    fast = True

    # Mapping of config keys to patterns that extract the number from CLAUDE.md prose.
    # Each entry: (config_key, config_file, list_of_regex_patterns)
    _CHECKS = [
        (
            "GATE_DIM",
            "base_config.yaml",
            [
                # "3 gate signals" or "1 gate signal"
                re.compile(r"(\d+)\s+gate\s+signals?", re.IGNORECASE),
                # "3D gate signal" or "1D gate signal"
                re.compile(r"(\d+)D\s+gate\s+signal", re.IGNORECASE),
                # MLPs (3→32→...) — the first number is gate dim
                re.compile(r"MLPs?\s*\((\d+)→\d+→"),
            ],
        ),
        (
            "NUM_TRANSFORMER_BLOCKS",
            "base_config.yaml",
            [
                # "6 `GatedTransformerBlock` layers" or "6 GatedTransformerBlock layers"
                re.compile(r"(\d+)\s+`?GatedTransformerBlock`?\s+layers?"),
            ],
        ),
    ]

    def _load_yaml(self, filename: str) -> dict:
        """Load a YAML config file, return {} on failure."""
        path = self.project_root / "plugins" / "tidal" / "configs" / filename
        if not path.exists():
            return {}
        if yaml is not None:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        # Fallback: simple key: value parsing for YAML without the library
        result = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    key, _, val = line.partition(":")
                    # Strip inline comments (# ...)
                    if "#" in val:
                        val = val[:val.index("#")]
                    val = val.strip().strip('"').strip("'")
                    try:
                        result[key.strip()] = int(val)
                    except ValueError:
                        try:
                            result[key.strip()] = float(val)
                        except ValueError:
                            result[key.strip()] = val
        return result

    def run(self) -> list[dict[str, Any]]:
        failures = []
        content = self._read_claude_md()
        lines = self._read_claude_md_lines()

        for config_key, config_file, patterns in self._CHECKS:
            config = self._load_yaml(config_file)
            if config_key not in config:
                continue
            expected = config[config_key]

            for pattern in patterns:
                for i, line in enumerate(lines, 1):
                    m = pattern.search(line)
                    if m:
                        found_val = int(m.group(1))
                        if found_val != expected:
                            failures.append(
                                self._failure(
                                    f"CLAUDE.md says {found_val} but {config_key}={expected} "
                                    f"in {config_file} (pattern: {pattern.pattern!r})",
                                    line=i,
                                )
                            )
        return failures


class ClassReferencesCheck(BaseCheck):
    """Class names in backticks -> grep for 'class ClassName' in Python files."""

    id = "class_references"
    severity = "warning"
    fast = False  # Requires grep, too slow for stop hook

    # Match PascalCase names in backticks (likely class names)
    _CLASS_RE = re.compile(r"`([A-Z][a-zA-Z0-9]+)`")

    # Known non-class names to skip
    _SKIP = {
        "CLAUDE", "MEMORY", "VOCAB_SIZE", "GATE_DIM", "HuggingFace",
        "TinyStories", "Redis", "Streamlit", "Docker", "Fastify", "React",
        "Vite", "BPE", "GPT", "AMP", "FFN", "MLP",
    }

    def run(self) -> list[dict[str, Any]]:
        failures = []
        content = self._read_claude_md()
        class_names = set(self._CLASS_RE.findall(content)) - self._SKIP

        for cls in sorted(class_names):
            # Search for "class ClassName" in Python files
            found = False
            for py_file in self.project_root.rglob("*.py"):
                # Skip venvs and node_modules
                parts = py_file.parts
                if any(
                    skip in parts
                    for skip in ("tidal-env", "node_modules", ".git", "__pycache__")
                ):
                    continue
                try:
                    text = py_file.read_text(errors="replace")
                    if f"class {cls}" in text:
                        found = True
                        break
                except OSError:
                    continue
            if not found:
                failures.append(
                    self._failure(f"Class `{cls}` not found in any Python file")
                )
        return failures


class MemoryLineCountCheck(BaseCheck):
    """MEMORY.md line count vs 160 threshold (200 is system limit)."""

    id = "memory_line_count"
    severity = "warning"
    fast = True

    THRESHOLD = 160

    def _find_memory_md(self) -> Path | None:
        """Search for MEMORY.md under .claude/projects/*/memory/."""
        claude_dir = self.project_root / ".claude" / "projects"
        if claude_dir.exists():
            for mem_file in claude_dir.rglob("MEMORY.md"):
                return mem_file
        # Also check the global user location
        home_claude = Path.home() / ".claude" / "projects"
        if home_claude.exists():
            for mem_file in home_claude.rglob("MEMORY.md"):
                return mem_file
        return None

    def run(self) -> list[dict[str, Any]]:
        mem_path = self._find_memory_md()
        if mem_path is None:
            return []
        try:
            line_count = len(mem_path.read_text().splitlines())
        except OSError:
            return []
        if line_count >= self.THRESHOLD:
            return [
                self._failure(
                    f"MEMORY.md has {line_count} lines (threshold: {self.THRESHOLD}, "
                    f"system limit: 200)"
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Validator engine
# ---------------------------------------------------------------------------

# Registry of all check classes
ALL_CHECKS: list[type[BaseCheck]] = [
    FileReferencesCheck,
    CommandEntrypointsCheck,
    DirectoryReferencesCheck,
    ArchitectureConstantsCheck,
    ClassReferencesCheck,
    MemoryLineCountCheck,
]


class MemoryValidator:
    """Orchestrates all checks and produces a structured report."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def run_all(self, fast: bool = False) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for check_cls in ALL_CHECKS:
            check = check_cls(self.project_root)
            if fast and not check.fast:
                continue
            try:
                results.extend(check.run())
            except Exception as e:
                results.append(
                    {
                        "check": check.id,
                        "severity": "warning",
                        "message": f"Check failed with error: {e}",
                        "line": None,
                    }
                )

        summary = {
            "critical": sum(1 for r in results if r["severity"] == "critical"),
            "warning": sum(1 for r in results if r["severity"] == "warning"),
        }
        return {"results": results, "summary": summary}


def compute_exit_code(report: dict[str, Any]) -> int:
    """0 = clean, 1 = critical failures, 2 = warnings only."""
    if report["summary"]["critical"] > 0:
        return 1
    if report["summary"]["warning"] > 0:
        return 2
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Validate CLAUDE.md and MEMORY.md")
    parser.add_argument("--fast", action="store_true", help="Fast mode (filesystem checks only)")
    parser.add_argument("--json", action="store_true", help="JSON output to stdout")
    parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Project root directory (default: git root or cwd)",
    )
    args = parser.parse_args()

    if args.project_dir:
        project_root = Path(args.project_dir).resolve()
    else:
        # Try to find git root
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                project_root = Path(result.stdout.strip())
            else:
                project_root = Path.cwd()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            project_root = Path.cwd()

    validator = MemoryValidator(project_root)
    report = validator.run_all(fast=args.fast)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        # Human-readable output
        for r in report["results"]:
            severity_tag = "CRITICAL" if r["severity"] == "critical" else "WARNING"
            line_info = f" (line {r['line']})" if r.get("line") else ""
            print(f"[{severity_tag}] [{r['check']}]{line_info}: {r['message']}")
        print()
        print(
            f"Summary: {report['summary']['critical']} critical, "
            f"{report['summary']['warning']} warnings"
        )

    sys.exit(compute_exit_code(report))


if __name__ == "__main__":
    main()
