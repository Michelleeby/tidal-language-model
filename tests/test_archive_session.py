"""Tests for the session archival hook (.claude/hooks/archive_session.py)."""

import json
import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

# Add the hook to the import path
HOOK_DIR = Path(__file__).resolve().parent.parent / ".claude" / "hooks"
sys.path.insert(0, str(HOOK_DIR))

import archive_session


def _make_real_format_lines():
    """Build JSONL lines matching the real Claude Code transcript format.

    Real transcripts nest role/content inside a ``message`` key and use
    ``type`` at the top level (``"user"`` or ``"assistant"``).
    """
    return [
        json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": "Hello, let's discuss gating signals in our model",
            },
            "uuid": "aaa",
        }),
        json.dumps({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "The gating mechanism modulates attention."},
                ],
            },
            "uuid": "bbb",
        }),
        json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": "Show me the reward curve",
            },
            "uuid": "ccc",
        }),
    ]


class TestParseTranscriptRealFormat(unittest.TestCase):
    """parse_transcript must handle the real Claude Code JSONL format where
    role and content are nested inside a ``message`` key."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.transcript = Path(self.tmp.name) / "session.jsonl"
        self.transcript.write_text("\n".join(_make_real_format_lines()) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_parses_user_messages(self):
        parsed = archive_session.parse_transcript(self.transcript)
        user_msgs = [m for m in parsed["messages"] if m["role"] == "user"]
        self.assertEqual(len(user_msgs), 2)
        self.assertIn("gating signals", user_msgs[0]["content"])

    def test_parses_assistant_messages(self):
        parsed = archive_session.parse_transcript(self.transcript)
        asst_msgs = [m for m in parsed["messages"] if m["role"] == "assistant"]
        self.assertEqual(len(asst_msgs), 1)
        self.assertIn("gating mechanism", asst_msgs[0]["content"])

    def test_derives_title_from_first_user_message(self):
        parsed = archive_session.parse_transcript(self.transcript)
        self.assertIn("gating signals", parsed["title"])

    def test_handles_tool_use_in_real_format(self):
        """Tool use blocks nested inside message.content should be captured."""
        lines = [
            json.dumps({
                "type": "user",
                "message": {"role": "user", "content": "Read the config file"},
                "uuid": "u1",
            }),
            json.dumps({
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me read that."},
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "/tmp/config.yaml"},
                        },
                    ],
                },
                "uuid": "a1",
            }),
        ]
        self.transcript.write_text("\n".join(lines) + "\n")
        parsed = archive_session.parse_transcript(self.transcript)
        self.assertEqual(len(parsed["tool_calls"]), 1)
        self.assertEqual(parsed["tool_calls"][0]["tool"], "Read")

    def test_skips_system_entries(self):
        """Entries with type 'system' should not produce messages."""
        lines = [
            json.dumps({
                "type": "system",
                "subtype": "local_command",
                "content": "<command-name>/skills</command-name>",
                "uuid": "s1",
            }),
        ] + _make_real_format_lines()
        self.transcript.write_text("\n".join(lines) + "\n")
        parsed = archive_session.parse_transcript(self.transcript)
        # Should only have 3 messages (2 user + 1 assistant), not the system one
        self.assertEqual(len(parsed["messages"]), 3)


class TestArchiveSessionUsesTranscriptPath(unittest.TestCase):
    """archive_session() should use an explicit transcript_path when provided,
    rather than searching the filesystem via find_latest_transcript()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.project_dir = self.tmp.name

        # Create a minimal JSONL transcript using real format (above the <2 threshold)
        self.transcript = Path(self.tmp.name) / "session.jsonl"
        self.transcript.write_text("\n".join(_make_real_format_lines()) + "\n")

        # Ensure output dirs exist
        (Path(self.project_dir) / "research" / "sessions").mkdir(parents=True)
        (Path(self.project_dir) / "research" / "data").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_uses_provided_transcript_path(self):
        """When transcript_path is given, archive_session uses it directly."""
        archive_session.archive_session(
            self.project_dir, transcript_path=str(self.transcript)
        )

        sessions_dir = Path(self.project_dir) / "research" / "sessions"
        md_files = list(sessions_dir.glob("*.md"))
        self.assertEqual(len(md_files), 1, "Should create exactly one session markdown")

    @patch("archive_session.find_latest_transcript")
    def test_does_not_call_find_when_transcript_path_given(self, mock_find):
        """find_latest_transcript should not be called when transcript_path is provided."""
        archive_session.archive_session(
            self.project_dir, transcript_path=str(self.transcript)
        )
        mock_find.assert_not_called()

    @patch("archive_session.find_latest_transcript")
    def test_falls_back_to_find_when_no_transcript_path(self, mock_find):
        """When transcript_path is None, should fall back to find_latest_transcript."""
        mock_find.return_value = self.transcript

        archive_session.archive_session(self.project_dir, transcript_path=None)

        mock_find.assert_called_once_with(self.project_dir)


class TestMainReadsTranscriptPath(unittest.TestCase):
    """main() should read transcript_path from the hook stdin JSON and pass it
    to archive_session()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.project_dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    @patch("archive_session.archive_session")
    def test_passes_transcript_path_from_stdin(self, mock_archive):
        hook_input = {
            "cwd": self.project_dir,
            "transcript_path": "/home/user/.claude/projects/abc/.history/sess.jsonl",
            "session_id": "sess-123",
        }
        with patch("sys.stdin", StringIO(json.dumps(hook_input))):
            with self.assertRaises(SystemExit) as ctx:
                archive_session.main()
            self.assertEqual(ctx.exception.code, 0)

        mock_archive.assert_called_once_with(
            self.project_dir,
            transcript_path="/home/user/.claude/projects/abc/.history/sess.jsonl",
        )

    @patch("archive_session.archive_session")
    def test_passes_none_when_transcript_path_absent(self, mock_archive):
        hook_input = {"cwd": self.project_dir}
        with patch("sys.stdin", StringIO(json.dumps(hook_input))):
            with self.assertRaises(SystemExit) as ctx:
                archive_session.main()
            self.assertEqual(ctx.exception.code, 0)

        mock_archive.assert_called_once_with(
            self.project_dir,
            transcript_path=None,
        )


class TestDebugLogging(unittest.TestCase):
    """The hook should write debug entries to a log file so the user can
    verify whether hooks actually fire."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.project_dir = self.tmp.name
        self.log_file = Path(self.project_dir) / ".claude" / "hooks" / "archive_debug.log"

        # Create transcript using real format
        self.transcript = Path(self.tmp.name) / "session.jsonl"
        self.transcript.write_text("\n".join(_make_real_format_lines()) + "\n")

        # Ensure output dirs
        (Path(self.project_dir) / "research" / "sessions").mkdir(parents=True)
        (Path(self.project_dir) / "research" / "data").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_creates_debug_log(self):
        """archive_session should write at least one line to the debug log."""
        archive_session.archive_session(
            self.project_dir, transcript_path=str(self.transcript)
        )

        self.assertTrue(self.log_file.exists(), "Debug log file should be created")
        content = self.log_file.read_text()
        self.assertGreater(len(content.strip()), 0, "Debug log should not be empty")

    def test_debug_log_contains_event_info(self):
        """Debug log should contain the transcript path for diagnostics."""
        archive_session.archive_session(
            self.project_dir, transcript_path=str(self.transcript)
        )

        content = self.log_file.read_text()
        self.assertIn(str(self.transcript), content)


class TestArchivedMarkerUsesTranscriptPath(unittest.TestCase):
    """The .archived marker should work correctly with an explicit transcript_path."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.project_dir = self.tmp.name

        self.transcript = Path(self.tmp.name) / "session.jsonl"
        self.transcript.write_text("\n".join(_make_real_format_lines()) + "\n")

        (Path(self.project_dir) / "research" / "sessions").mkdir(parents=True)
        (Path(self.project_dir) / "research" / "data").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_second_call_skips_already_archived(self):
        """If the transcript is already archived, a second call should not create a duplicate."""
        archive_session.archive_session(
            self.project_dir, transcript_path=str(self.transcript)
        )
        archive_session.archive_session(
            self.project_dir, transcript_path=str(self.transcript)
        )

        sessions_dir = Path(self.project_dir) / "research" / "sessions"
        md_files = list(sessions_dir.glob("*.md"))
        self.assertEqual(len(md_files), 1, "Should not create duplicate archives")


if __name__ == "__main__":
    unittest.main()
