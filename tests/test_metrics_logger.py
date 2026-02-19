"""Tests for MetricsLogger configurable Redis prefix."""

import os
import json
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock

from MetricsLogger import MetricsLogger


class TestMetricsLoggerRedisPrefix(unittest.TestCase):
    """Verify that MetricsLogger uses configurable Redis key prefixes."""

    def _make_logger(self, tmpdir, redis_prefix=None):
        """Create a MetricsLogger with Redis disabled and custom prefix."""
        with patch.object(MetricsLogger, "_init_redis"):
            logger = MetricsLogger(tmpdir, redis_prefix=redis_prefix)
            logger._redis = None
            logger._redis_available = False
            return logger

    def test_default_prefix_is_tidal(self):
        """When no redis_prefix is given and no env var, it defaults to 'tidal'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("METRICS_REDIS_PREFIX", None)
                ml = self._make_logger(tmpdir)
                self.assertEqual(ml._redis_prefix, "tidal")

    def test_reads_prefix_from_env_var(self):
        """When METRICS_REDIS_PREFIX is set and no explicit prefix, uses env var."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"METRICS_REDIS_PREFIX": "envprefix"}, clear=False):
                ml = self._make_logger(tmpdir)
                self.assertEqual(ml._redis_prefix, "envprefix")

    def test_custom_prefix(self):
        """Constructor stores the given redis_prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = self._make_logger(tmpdir, redis_prefix="mymodel")
            self.assertEqual(ml._redis_prefix, "mymodel")

    def test_status_redis_key_uses_prefix(self):
        """_update_status should write to '{prefix}:status:{expId}'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = self._make_logger(tmpdir, redis_prefix="custom")
            ml._redis_available = True
            ml._redis = MagicMock()
            ml._http_enabled = False

            ml._update_status({"status": "training"})

            exp_id = os.path.basename(tmpdir)
            ml._redis.set.assert_any_call(
                f"custom:status:{exp_id}",
                json.dumps({"status": "training"}),
                ex=900,
            )
            ml._redis.sadd.assert_called_with(f"custom:experiments", exp_id)

    def test_log_metrics_redis_keys_use_prefix(self):
        """log_metrics should write to '{prefix}:metrics:{expId}:*' keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = self._make_logger(tmpdir, redis_prefix="abc")
            mock_pipe = MagicMock()
            ml._redis_available = True
            ml._redis = MagicMock()
            ml._redis.pipeline.return_value = mock_pipe
            ml._http_enabled = False

            ml.log_metrics({"loss": 1.5}, step=10)

            exp_id = os.path.basename(tmpdir)
            # Check pipeline calls used the custom prefix
            mock_pipe.set.assert_called_once()
            args = mock_pipe.set.call_args[0]
            self.assertTrue(args[0].startswith(f"abc:metrics:{exp_id}:latest"))

            mock_pipe.rpush.assert_called_once()
            rpush_args = mock_pipe.rpush.call_args[0]
            self.assertTrue(rpush_args[0].startswith(f"abc:metrics:{exp_id}:history"))

    def test_log_rl_metrics_redis_key_uses_prefix(self):
        """log_rl_metrics should write to '{prefix}:rl:{expId}:latest'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = self._make_logger(tmpdir, redis_prefix="xyz")
            ml._redis_available = True
            ml._redis = MagicMock()
            ml._http_enabled = False

            ml.log_rl_metrics({"episode_rewards": [1.0]}, global_step=5)

            exp_id = os.path.basename(tmpdir)
            ml._redis.set.assert_called()
            # Find the rl:latest call (not the status call)
            rl_calls = [
                c for c in ml._redis.set.call_args_list
                if "rl:" in str(c)
            ]
            self.assertTrue(len(rl_calls) > 0)
            self.assertTrue(rl_calls[0][0][0].startswith(f"xyz:rl:{exp_id}:latest"))

    def test_env_var_fallback_for_http(self):
        """MetricsLogger reads TRAINING_* env vars with TIDAL_* fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "TRAINING_API_URL": "http://example.com",
                "TRAINING_AUTH_TOKEN": "tok123",
                "TRAINING_JOB_ID": "job-42",
            }
            with patch.dict(os.environ, env, clear=False):
                ml = self._make_logger(tmpdir)
                self.assertEqual(ml._http_url, "http://example.com")
                self.assertEqual(ml._http_token, "tok123")
                self.assertEqual(ml._http_job_id, "job-42")
                self.assertTrue(ml._http_enabled)


class TestMetricsLoggerUsesRequests(unittest.TestCase):
    """Verify MetricsLogger uses requests.Session instead of urllib."""

    def _make_http_logger(self, tmpdir):
        """Create a MetricsLogger with HTTP enabled and a mock session."""
        env = {
            "TRAINING_API_URL": "http://example.com",
            "TRAINING_AUTH_TOKEN": "tok123",
            "TRAINING_JOB_ID": "job-42",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch.object(MetricsLogger, "_init_redis"):
                ml = MetricsLogger(tmpdir)
                ml._redis = None
                ml._redis_available = False
                return ml

    def test_has_session(self):
        """MetricsLogger creates a requests.Session when HTTP is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = self._make_http_logger(tmpdir)
            self.assertIsNotNone(ml._http_session)
            self.assertEqual(
                ml._http_session.headers["Authorization"], "Bearer tok123",
            )

    def test_http_forward_uses_session_post(self):
        """_http_forward delegates to session.post instead of urllib."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = self._make_http_logger(tmpdir)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()

            with patch.object(ml._http_session, "post", return_value=mock_resp) as mock_post:
                ml._http_forward({"expId": "test-exp", "points": []})

            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            self.assertIn("/api/workers/job-42/metrics", args[0])
            self.assertEqual(kwargs["json"], {"expId": "test-exp", "points": []})

    def test_upload_single_uses_session_put(self):
        """_upload_single uses session.put with file streaming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = self._make_http_logger(tmpdir)

            filepath = os.path.join(tmpdir, "model.pth")
            with open(filepath, "wb") as f:
                f.write(b"fake model data")

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = '{"ok":true}'

            with patch.object(ml._http_session, "put", return_value=mock_resp) as mock_put:
                ml._upload_single(filepath, "model.pth", 15, retries=1)

            mock_put.assert_called_once()
            args, kwargs = mock_put.call_args
            self.assertIn("model.pth", args[0])
            self.assertEqual(kwargs["headers"]["Content-Type"], "application/octet-stream")

    def test_report_experiment_id_uses_session_patch(self):
        """_report_experiment_id uses session.patch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = self._make_http_logger(tmpdir)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()

            with patch.object(ml._http_session, "patch", return_value=mock_resp) as mock_patch:
                ml._report_experiment_id()

            mock_patch.assert_called_once()
            args, kwargs = mock_patch.call_args
            self.assertIn("/experiment-id", args[0])


class TestMetricsLoggerUploadThreadTracking(unittest.TestCase):
    """Verify that MetricsLogger tracks and joins upload threads."""

    def _make_http_logger(self, tmpdir):
        """Create a MetricsLogger with HTTP enabled and a mock session."""
        env = {
            "TRAINING_API_URL": "http://example.com",
            "TRAINING_AUTH_TOKEN": "tok123",
            "TRAINING_JOB_ID": "job-42",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch.object(MetricsLogger, "_init_redis"):
                ml = MetricsLogger(tmpdir)
                ml._redis = None
                ml._redis_available = False
                return ml

    def test_upload_threads_tracked(self):
        """upload_checkpoint appends thread to _upload_threads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = self._make_http_logger(tmpdir)

            # Create a fake checkpoint file
            filepath = os.path.join(tmpdir, "checkpoint.pth")
            with open(filepath, "wb") as f:
                f.write(b"fake data")

            # Patch _upload_file to avoid real uploads
            with patch.object(ml, "_upload_file"):
                ml.upload_checkpoint(filepath)

            self.assertEqual(len(ml._upload_threads), 1)
            self.assertIsInstance(ml._upload_threads[0], threading.Thread)

    def test_finalize_joins_upload_threads(self):
        """finalize() calls .join() on all pending upload threads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = self._make_http_logger(tmpdir)

            # Create mock threads that simulate in-progress uploads
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = True
            ml._upload_threads = [mock_thread]

            ml.finalize()

            mock_thread.join.assert_called_once()

    def test_completed_threads_reaped(self):
        """Completed threads are removed from _upload_threads on next upload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = self._make_http_logger(tmpdir)

            # Add a completed (dead) thread
            dead_thread = MagicMock()
            dead_thread.is_alive.return_value = False
            ml._upload_threads = [dead_thread]

            # Create a fake checkpoint and trigger another upload
            filepath = os.path.join(tmpdir, "checkpoint2.pth")
            with open(filepath, "wb") as f:
                f.write(b"fake data")

            with patch.object(ml, "_upload_file"):
                ml.upload_checkpoint(filepath)

            # Dead thread should be reaped; only the new thread remains
            self.assertEqual(len(ml._upload_threads), 1)
            self.assertIsNot(ml._upload_threads[0], dead_thread)


if __name__ == "__main__":
    unittest.main()
