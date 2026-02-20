"""Tests for experiment_utils — shared utility functions."""

import hashlib
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import torch

from experiment_utils import (
    get_git_commit_hash,
    get_file_hash,
    resolve_device,
    report_experiment_id_to_job,
    create_experiment_metadata,
    write_experiment_metadata,
    read_experiment_metadata,
)


class TestGetGitCommitHash(unittest.TestCase):
    """Tests for get_git_commit_hash()."""

    def test_returns_string(self):
        result = get_git_commit_hash()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_returns_short_hash_in_git_repo(self):
        # We're running inside a git repo, so this should return a real hash
        result = get_git_commit_hash()
        self.assertNotEqual(result, "nogit")
        # Short hash is typically 7-12 chars
        self.assertLessEqual(len(result), 12)

    @patch("experiment_utils.subprocess.check_output", side_effect=FileNotFoundError)
    def test_returns_nogit_when_git_missing(self, _mock):
        result = get_git_commit_hash()
        self.assertEqual(result, "nogit")

    @patch("experiment_utils.subprocess.check_output")
    def test_returns_nogit_on_subprocess_error(self, mock_check):
        import subprocess
        mock_check.side_effect = subprocess.CalledProcessError(1, "git")
        result = get_git_commit_hash()
        self.assertEqual(result, "nogit")


class TestGetFileHash(unittest.TestCase):
    """Tests for get_file_hash()."""

    def test_returns_10_char_hex_string(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\n")
            f.flush()
            result = get_file_hash(f.name)
        os.unlink(f.name)
        self.assertEqual(len(result), 10)
        # Must be valid hex
        int(result, 16)

    def test_deterministic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("test content\n")
            f.flush()
            r1 = get_file_hash(f.name)
            r2 = get_file_hash(f.name)
        os.unlink(f.name)
        self.assertEqual(r1, r2)

    def test_different_files_different_hashes(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            f1.write("content A\n")
            f1.flush()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            f2.write("content B\n")
            f2.flush()

        h1 = get_file_hash(f1.name)
        h2 = get_file_hash(f2.name)
        os.unlink(f1.name)
        os.unlink(f2.name)
        self.assertNotEqual(h1, h2)

    def test_matches_manual_sha256(self):
        content = b"hello world\n"
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            result = get_file_hash(f.name)

        expected = hashlib.sha256(content).hexdigest()[:10]
        os.unlink(f.name)
        self.assertEqual(result, expected)


class TestResolveDevice(unittest.TestCase):
    """Tests for resolve_device()."""

    def test_auto_returns_valid_device(self):
        device = resolve_device({"DEVICE": "auto"})
        self.assertIsInstance(device, torch.device)
        # Must be either cpu or cuda
        self.assertIn(device.type, ("cpu", "cuda"))

    def test_defaults_to_auto_when_key_missing(self):
        device = resolve_device({})
        self.assertIsInstance(device, torch.device)

    def test_explicit_cpu(self):
        device = resolve_device({"DEVICE": "cpu"})
        self.assertEqual(device, torch.device("cpu"))

    def test_explicit_cuda_string(self):
        # This should create the device object regardless of hardware
        device = resolve_device({"DEVICE": "cpu"})
        self.assertEqual(device.type, "cpu")


class TestReportExperimentIdToJob(unittest.TestCase):
    """Tests for report_experiment_id_to_job()."""

    def test_noop_when_no_job_id(self):
        """Should do nothing if TIDAL_JOB_ID is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TIDAL_JOB_ID", None)
            # Should not raise
            report_experiment_id_to_job("exp-123")

    @patch("experiment_utils.redis_lib")
    def test_updates_redis_when_job_id_set(self, mock_redis_mod):
        """Should update Redis job record when TIDAL_JOB_ID is set."""
        import json

        mock_r = MagicMock()
        mock_redis_mod.from_url.return_value = mock_r
        mock_r.hget.return_value = json.dumps({
            "jobId": "job-abc",
            "status": "running",
        })

        with patch.dict(os.environ, {"TIDAL_JOB_ID": "job-abc"}):
            report_experiment_id_to_job("exp-456")

        # Should have called hset with updated data
        mock_r.hset.assert_called_once()
        call_args = mock_r.hset.call_args
        stored = json.loads(call_args[0][2])
        self.assertEqual(stored["experimentId"], "exp-456")
        self.assertIn("updatedAt", stored)

        # Should have published notification
        mock_r.publish.assert_called_once()

    @patch("experiment_utils.redis_lib")
    def test_silent_on_redis_error(self, mock_redis_mod):
        """Should silently handle Redis errors."""
        mock_redis_mod.from_url.side_effect = Exception("Connection refused")

        with patch.dict(os.environ, {"TIDAL_JOB_ID": "job-abc"}):
            # Should not raise
            report_experiment_id_to_job("exp-789")

    @patch("experiment_utils.redis_lib")
    def test_noop_when_job_not_found_in_redis(self, mock_redis_mod):
        """Should handle case where job record doesn't exist in Redis."""
        mock_r = MagicMock()
        mock_redis_mod.from_url.return_value = mock_r
        mock_r.hget.return_value = None

        with patch.dict(os.environ, {"TIDAL_JOB_ID": "job-missing"}):
            report_experiment_id_to_job("exp-000")

        mock_r.hset.assert_not_called()


class TestCreateExperimentMetadata(unittest.TestCase):
    """Tests for create_experiment_metadata()."""

    def test_lm_metadata_has_required_fields(self):
        meta = create_experiment_metadata("lm")
        self.assertEqual(meta["type"], "lm")
        self.assertIn("created_at", meta)
        self.assertIsNone(meta.get("source_experiment_id"))
        self.assertIsNone(meta.get("source_checkpoint"))

    def test_rl_metadata_includes_source_fields(self):
        meta = create_experiment_metadata(
            "rl",
            source_experiment_id="20250101-commit_abc-config_def",
            source_checkpoint="experiments/20250101-commit_abc-config_def/model.pth",
        )
        self.assertEqual(meta["type"], "rl")
        self.assertEqual(meta["source_experiment_id"], "20250101-commit_abc-config_def")
        self.assertEqual(
            meta["source_checkpoint"],
            "experiments/20250101-commit_abc-config_def/model.pth",
        )
        self.assertIn("created_at", meta)

    def test_created_at_is_iso_string(self):
        meta = create_experiment_metadata("lm")
        # Should be parseable as an ISO timestamp
        from datetime import datetime
        datetime.fromisoformat(meta["created_at"])

    def test_rejects_invalid_type(self):
        with self.assertRaises(ValueError):
            create_experiment_metadata("invalid")


class TestWriteAndReadExperimentMetadata(unittest.TestCase):
    """Tests for write_experiment_metadata() and read_experiment_metadata()."""

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = create_experiment_metadata("lm")
            write_experiment_metadata(tmpdir, meta)
            loaded = read_experiment_metadata(tmpdir)
            self.assertEqual(loaded, meta)

    def test_writes_metadata_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = create_experiment_metadata("rl", source_experiment_id="src-exp")
            write_experiment_metadata(tmpdir, meta)
            metadata_path = os.path.join(tmpdir, "metadata.json")
            self.assertTrue(os.path.exists(metadata_path))

    def test_read_returns_none_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = read_experiment_metadata(tmpdir)
            self.assertIsNone(result)

    def test_read_returns_none_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = os.path.join(tmpdir, "metadata.json")
            with open(bad_path, "w") as f:
                f.write("not valid json {{{")
            result = read_experiment_metadata(tmpdir)
            self.assertIsNone(result)

    def test_rl_metadata_round_trip_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = create_experiment_metadata(
                "rl",
                source_experiment_id="exp-abc",
                source_checkpoint="/path/to/model.pth",
            )
            write_experiment_metadata(tmpdir, meta)
            loaded = read_experiment_metadata(tmpdir)
            self.assertEqual(loaded["type"], "rl")
            self.assertEqual(loaded["source_experiment_id"], "exp-abc")
            self.assertEqual(loaded["source_checkpoint"], "/path/to/model.pth")


class TestGetPreassignedExperimentId(unittest.TestCase):
    """Tests for get_preassigned_experiment_id()."""

    def test_returns_env_var_when_set(self):
        """Should return the TIDAL_EXPERIMENT_ID value when the env var is set."""
        with patch.dict(os.environ, {"TIDAL_EXPERIMENT_ID": "pre-assigned-exp-123"}):
            from experiment_utils import get_preassigned_experiment_id
            result = get_preassigned_experiment_id()
            self.assertEqual(result, "pre-assigned-exp-123")

    def test_returns_none_when_not_set(self):
        """Should return None when TIDAL_EXPERIMENT_ID is not in environment."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TIDAL_EXPERIMENT_ID", None)
            from experiment_utils import get_preassigned_experiment_id
            result = get_preassigned_experiment_id()
            self.assertIsNone(result)

    def test_returns_none_for_empty_string(self):
        """Should return None when TIDAL_EXPERIMENT_ID is an empty string."""
        with patch.dict(os.environ, {"TIDAL_EXPERIMENT_ID": ""}):
            from experiment_utils import get_preassigned_experiment_id
            result = get_preassigned_experiment_id()
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
