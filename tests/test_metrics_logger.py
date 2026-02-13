"""Tests for MetricsLogger configurable Redis prefix."""

import os
import json
import tempfile
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


if __name__ == "__main__":
    unittest.main()
