"""Tests for WorkerAgent: checkpoint downloads, requests migration, and log streaming."""

import io
import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call

from worker_agent import WorkerAgent, RedisTransport, HttpTransport


class FakeRequestsResponse:
    """Minimal requests.Response stub for _download_checkpoint tests."""

    def __init__(self, status_code, body=b"", headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = ""

    def iter_content(self, chunk_size=None):
        if self._body:
            yield self._body

    def raise_for_status(self):
        pass


def _make_agent(transport):
    """Create a WorkerAgent with signal handlers suppressed."""
    with patch("signal.signal"):
        return WorkerAgent("job-123", transport)


def _make_http_transport(api_url="http://localhost:4400", token="tok"):
    return HttpTransport(api_url, token)


def _make_redis_transport():
    with patch("worker_agent.redis_lib"):
        return RedisTransport("redis://localhost:6379")


# ---------------------------------------------------------------------------
# _download_checkpoint guard clauses
# ---------------------------------------------------------------------------

class TestDownloadCheckpointGuards(unittest.TestCase):
    """Tests for early-return guard clauses in _download_checkpoint."""

    def test_skips_when_transport_is_redis(self):
        """RedisTransport means local job — checkpoint already on disk."""
        transport = _make_redis_transport()
        agent = _make_agent(transport)
        # Should return without error even with no checkpoint key
        agent._download_checkpoint({"type": "rl-training"})

    def test_skips_when_no_checkpoint_in_config(self):
        """No checkpoint key means the job doesn't need one (e.g. LM training)."""
        transport = _make_http_transport()
        agent = _make_agent(transport)
        agent._download_checkpoint({"type": "rl-training"})

    def test_skips_when_checkpoint_file_already_exists(self):
        """If the file is already on disk, don't re-download."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            exp_dir = os.path.join(tmpdir, "experiments", "exp-abc")
            os.makedirs(exp_dir)
            ckpt_path = os.path.join(exp_dir, "model.pth")
            with open(ckpt_path, "wb") as f:
                f.write(b"fake model data")

            # Should return without making any HTTP calls
            with patch.object(transport.session, "get") as mock_get:
                agent._download_checkpoint({
                    "checkpoint": "experiments/exp-abc/model.pth",
                })
                mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# _download_checkpoint path parsing + directory creation
# ---------------------------------------------------------------------------

class TestDownloadCheckpointPathParsing(unittest.TestCase):
    """Tests for path parsing and directory creation."""

    def test_parses_checkpoint_path_correctly(self):
        """Extracts expId and filename from 'experiments/<expId>/<filename>'."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            body = b"model weights"

            fake_resp = FakeRequestsResponse(200, body, {
                "content-length": str(len(body)),
            })

            with patch.object(transport.session, "get", return_value=fake_resp) as mock_get:
                agent._download_checkpoint({
                    "checkpoint": "experiments/abc-123/transformer-lm_v1.0.0.pth",
                })

            # Verify the request URL includes expId and filename
            mock_get.assert_called_once()
            req_url = mock_get.call_args[0][0]
            self.assertIn("transformer-lm_v1.0.0.pth", req_url)
            self.assertIn("expId=abc-123", req_url)

    def test_parses_absolute_checkpoint_path(self):
        """Handles absolute paths like '/data/experiments/<expId>/<filename>'."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            body = b"model weights"

            fake_resp = FakeRequestsResponse(200, body, {
                "content-length": str(len(body)),
            })

            with patch.object(transport.session, "get", return_value=fake_resp):
                agent._download_checkpoint({
                    "checkpoint": "/data/experiments/20260213-exp-abc/checkpoint_epoch_5.pth",
                })

            # File should be saved locally under project_root/experiments/
            final = os.path.join(tmpdir, "experiments", "20260213-exp-abc", "checkpoint_epoch_5.pth")
            self.assertTrue(os.path.exists(final))

    def test_rewrites_config_checkpoint_to_local_path(self):
        """After download, config['checkpoint'] is rewritten to a local relative path."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            body = b"model data"

            fake_resp = FakeRequestsResponse(200, body, {
                "content-length": str(len(body)),
            })

            config = {
                "checkpoint": "/data/experiments/exp-xyz/model.pth",
            }
            with patch.object(transport.session, "get", return_value=fake_resp):
                agent._download_checkpoint(config)

            self.assertEqual(
                config["checkpoint"],
                os.path.join("experiments", "exp-xyz", "model.pth"),
            )

    def test_rewrites_config_when_file_already_exists(self):
        """Config path is rewritten even when skipping download (file exists)."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            exp_dir = os.path.join(tmpdir, "experiments", "exp-abc")
            os.makedirs(exp_dir)
            with open(os.path.join(exp_dir, "model.pth"), "wb") as f:
                f.write(b"existing")

            config = {
                "checkpoint": "/data/experiments/exp-abc/model.pth",
            }
            agent._download_checkpoint(config)

            self.assertEqual(
                config["checkpoint"],
                os.path.join("experiments", "exp-abc", "model.pth"),
            )

    def test_creates_experiment_directory(self):
        """Creates experiments/<expId>/ if it doesn't exist."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            body = b"model data"

            fake_resp = FakeRequestsResponse(200, body, {
                "content-length": str(len(body)),
            })

            with patch.object(transport.session, "get", return_value=fake_resp):
                agent._download_checkpoint({
                    "checkpoint": "experiments/new-exp/model.pth",
                })

            exp_dir = os.path.join(tmpdir, "experiments", "new-exp")
            self.assertTrue(os.path.isdir(exp_dir))


# ---------------------------------------------------------------------------
# _download_checkpoint streaming + atomic rename
# ---------------------------------------------------------------------------

class TestDownloadCheckpointStreaming(unittest.TestCase):
    """Tests for streaming download, temp file, and atomic rename."""

    def test_writes_to_tmp_then_renames(self):
        """File is written to .tmp first, then atomically renamed."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            body = b"x" * 1024

            fake_resp = FakeRequestsResponse(200, body, {
                "content-length": str(len(body)),
            })

            with patch.object(transport.session, "get", return_value=fake_resp):
                agent._download_checkpoint({
                    "checkpoint": "experiments/exp-1/model.pth",
                })

            final_path = os.path.join(tmpdir, "experiments", "exp-1", "model.pth")
            tmp_path = final_path + ".tmp"

            # Final file should exist with correct content
            self.assertTrue(os.path.exists(final_path))
            with open(final_path, "rb") as f:
                self.assertEqual(f.read(), body)

            # Temp file should NOT exist after successful download
            self.assertFalse(os.path.exists(tmp_path))

    def test_verifies_content_length(self):
        """Raises RuntimeError if downloaded bytes don't match Content-Length."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir

            # Server says 1000 bytes but only sends 500
            fake_resp = FakeRequestsResponse(200, b"x" * 500, {
                "content-length": "1000",
            })

            with patch.object(transport.session, "get", return_value=fake_resp):
                with self.assertRaises(RuntimeError) as ctx:
                    agent._download_checkpoint({
                        "checkpoint": "experiments/exp-1/model.pth",
                    })
                self.assertIn("Content-Length", str(ctx.exception))


# ---------------------------------------------------------------------------
# _download_checkpoint retry + error handling
# ---------------------------------------------------------------------------

class TestDownloadCheckpointRetry(unittest.TestCase):
    """Tests for retry logic and error handling."""

    @patch("time.sleep")  # Don't actually sleep in tests
    def test_retries_on_5xx(self, mock_sleep):
        """Retries up to 3 times on 5xx responses."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            body = b"model data"

            fail_resp = FakeRequestsResponse(502, b"Bad Gateway")
            ok_resp = FakeRequestsResponse(200, body, {
                "content-length": str(len(body)),
            })

            with patch.object(transport.session, "get", side_effect=[fail_resp, fail_resp, ok_resp]):
                agent._download_checkpoint({
                    "checkpoint": "experiments/exp-1/model.pth",
                })

            # Should have succeeded on third attempt
            final_path = os.path.join(tmpdir, "experiments", "exp-1", "model.pth")
            self.assertTrue(os.path.exists(final_path))
            # Should have slept twice (exponential backoff)
            self.assertEqual(mock_sleep.call_count, 2)

    def test_raises_on_404(self):
        """404 means checkpoint not on server — immediate RuntimeError, no retry."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir

            fake_resp = FakeRequestsResponse(404, b"Not found")

            with patch.object(transport.session, "get", return_value=fake_resp) as mock_get:
                with self.assertRaises(RuntimeError) as ctx:
                    agent._download_checkpoint({
                        "checkpoint": "experiments/exp-1/model.pth",
                    })
                self.assertIn("404", str(ctx.exception))

            # Should NOT have retried
            self.assertEqual(mock_get.call_count, 1)

    @patch("time.sleep")
    def test_raises_after_all_retries_exhausted(self, mock_sleep):
        """Raises RuntimeError after 3 consecutive 5xx failures."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir

            fail_resp_1 = FakeRequestsResponse(503, b"Unavailable")
            fail_resp_2 = FakeRequestsResponse(503, b"Unavailable")
            fail_resp_3 = FakeRequestsResponse(503, b"Unavailable")

            with patch.object(transport.session, "get", side_effect=[fail_resp_1, fail_resp_2, fail_resp_3]):
                with self.assertRaises(RuntimeError) as ctx:
                    agent._download_checkpoint({
                        "checkpoint": "experiments/exp-1/model.pth",
                    })
                self.assertIn("after 3 attempts", str(ctx.exception))

    @patch("time.sleep")
    def test_retries_on_connection_error(self, mock_sleep):
        """Retries on connection errors (not just HTTP 5xx)."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            body = b"model data"

            ok_resp = FakeRequestsResponse(200, body, {
                "content-length": str(len(body)),
            })

            with patch.object(transport.session, "get", side_effect=[
                ConnectionError("refused"),
                ok_resp,
            ]):
                agent._download_checkpoint({
                    "checkpoint": "experiments/exp-1/model.pth",
                })

            final_path = os.path.join(tmpdir, "experiments", "exp-1", "model.pth")
            self.assertTrue(os.path.exists(final_path))

    def test_cleans_up_tmp_on_error(self):
        """Temp file is removed if download fails."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir

            fake_resp = FakeRequestsResponse(404, b"Not found")

            with patch.object(transport.session, "get", return_value=fake_resp):
                with self.assertRaises(RuntimeError):
                    agent._download_checkpoint({
                        "checkpoint": "experiments/exp-1/model.pth",
                    })

            # No .tmp file should remain
            exp_dir = os.path.join(tmpdir, "experiments", "exp-1")
            if os.path.isdir(exp_dir):
                tmp_files = [f for f in os.listdir(exp_dir) if f.endswith(".tmp")]
                self.assertEqual(tmp_files, [])


# ---------------------------------------------------------------------------
# run() integration — download before running
# ---------------------------------------------------------------------------

class TestRunDownloadIntegration(unittest.TestCase):
    """Tests that run() calls _download_checkpoint at the right time."""

    def test_calls_download_before_update_status_running(self):
        """_download_checkpoint is called BEFORE update_status('running')."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        call_order = []

        job = {
            "config": {
                "type": "rl-training",
                "configPath": "configs/base_config.yaml",
                "checkpoint": "experiments/exp-1/model.pth",
            },
        }

        transport.get_job = MagicMock(return_value=job)
        transport.update_status = MagicMock(
            side_effect=lambda *a, **kw: call_order.append(("update_status", a[1]))
        )
        transport.send_heartbeat = MagicMock()
        transport.read_signal = MagicMock(return_value=None)

        def fake_download(config):
            call_order.append(("download", None))

        with patch.object(agent, "_download_checkpoint", side_effect=fake_download):
            with patch.object(agent, "_run_training", return_value=0):
                with patch.object(agent, "_final_checkpoint_sweep"):
                    agent.run()

        # download must come before the first update_status("running")
        self.assertEqual(call_order[0], ("download", None))
        self.assertEqual(call_order[1], ("update_status", "running"))

    def test_reports_failed_on_download_error(self):
        """run() reports 'failed' status if _download_checkpoint raises."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        job = {
            "config": {
                "type": "rl-training",
                "configPath": "configs/base_config.yaml",
                "checkpoint": "experiments/exp-1/model.pth",
            },
        }

        transport.get_job = MagicMock(return_value=job)
        transport.update_status = MagicMock()

        def boom(config):
            raise RuntimeError("Checkpoint not found on server (404)")

        with patch.object(agent, "_download_checkpoint", side_effect=boom):
            agent.run()

        # Should have reported failed status
        transport.update_status.assert_called_once()
        args = transport.update_status.call_args
        self.assertEqual(args[0][1], "failed")
        self.assertIn("404", args[1]["error"] if "error" in (args[1] or {}) else args[0][2])

    def test_does_not_spawn_training_on_download_failure(self):
        """Training subprocess must NOT be spawned if download fails."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        job = {
            "config": {
                "type": "rl-training",
                "configPath": "configs/base_config.yaml",
                "checkpoint": "experiments/exp-1/model.pth",
            },
        }

        transport.get_job = MagicMock(return_value=job)
        transport.update_status = MagicMock()

        with patch.object(agent, "_download_checkpoint", side_effect=RuntimeError("fail")):
            with patch.object(agent, "_run_training") as mock_train:
                agent.run()
                mock_train.assert_not_called()


# ---------------------------------------------------------------------------
# Heartbeat exception logging
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Subprocess environment variable allowlist
# ---------------------------------------------------------------------------

class TestSubprocessEnvAllowlist(unittest.TestCase):
    """Tests that _spawn_and_monitor filters env vars instead of leaking all."""

    def _get_constructed_env(self, extra_environ=None):
        """Build the env dict that _spawn_and_monitor would pass to Popen."""
        transport = _make_http_transport()
        agent = _make_agent(transport)
        transport.send_logs = MagicMock()
        transport.read_signal = MagicMock(return_value=None)

        original_environ = os.environ.copy()
        if extra_environ:
            os.environ.update(extra_environ)

        try:
            with patch("subprocess.Popen") as mock_popen:
                mock_proc = MagicMock()
                mock_proc.poll.return_value = 0
                mock_proc.pid = 12345
                mock_proc.stdout = io.BytesIO(b"")
                mock_proc.stderr = io.BytesIO(b"")
                mock_popen.return_value = mock_proc

                agent._heartbeat_stop = threading.Event()
                agent._heartbeat_stop.set()

                agent._spawn_and_monitor(["echo", "test"], redis_prefix="tidal")

                # Extract the env kwarg passed to Popen
                call_kwargs = mock_popen.call_args
                return call_kwargs[1].get("env") or call_kwargs.kwargs.get("env")
        finally:
            os.environ.clear()
            os.environ.update(original_environ)

    def test_subprocess_env_excludes_sensitive_vars(self):
        """Sensitive vars like AWS_SECRET_KEY should NOT be in subprocess env."""
        env = self._get_constructed_env({"AWS_SECRET_KEY": "supersecret123"})
        self.assertNotIn("AWS_SECRET_KEY", env)

    def test_subprocess_env_includes_required_vars(self):
        """PATH and TRAINING_JOB_ID must be present."""
        env = self._get_constructed_env()
        self.assertIn("PATH", env)
        self.assertIn("TRAINING_JOB_ID", env)

    def test_subprocess_env_passes_job_vars(self):
        """TRAINING_JOB_ID and METRICS_REDIS_PREFIX are set correctly."""
        env = self._get_constructed_env()
        self.assertEqual(env["TRAINING_JOB_ID"], "job-123")
        self.assertEqual(env["METRICS_REDIS_PREFIX"], "tidal")

    def test_subprocess_env_passes_cuda_vars(self):
        """CUDA_ prefixed vars should pass through."""
        env = self._get_constructed_env({"CUDA_VISIBLE_DEVICES": "0,1"})
        self.assertEqual(env.get("CUDA_VISIBLE_DEVICES"), "0,1")


class TestHeartbeatExceptionLogging(unittest.TestCase):
    """Tests that heartbeat thread logs exceptions instead of silently swallowing."""

    @patch("worker_agent.HEARTBEAT_INTERVAL", 0.01)
    def test_heartbeat_logs_exceptions(self):
        """When send_heartbeat raises, exception should be printed to stderr."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        # Make send_heartbeat raise
        transport.send_heartbeat = MagicMock(side_effect=ConnectionError("refused"))

        agent._heartbeat_stop = threading.Event()

        captured = io.StringIO()

        with patch("sys.stderr", captured):
            agent._start_heartbeat()
            # Let the heartbeat thread run a few cycles
            time.sleep(0.1)
            agent._heartbeat_stop.set()
            time.sleep(0.05)

        stderr_output = captured.getvalue()
        self.assertIn("Heartbeat failed", stderr_output)
        self.assertIn("refused", stderr_output)


# ---------------------------------------------------------------------------
# HttpTransport uses requests.Session
# ---------------------------------------------------------------------------

class TestHttpTransportRequests(unittest.TestCase):
    """Tests that HttpTransport uses requests.Session for HTTP calls."""

    def test_init_creates_session(self):
        """HttpTransport.__init__ creates a requests.Session with auth headers."""
        transport = HttpTransport("https://example.com", "tok-123")
        self.assertIsNotNone(transport.session)
        self.assertEqual(
            transport.session.headers["Authorization"], "Bearer tok-123",
        )
        self.assertEqual(
            transport.session.headers["User-Agent"], "TidalWorker/1.0",
        )

    def test_request_uses_session(self):
        """_request delegates to self.session.request."""
        transport = HttpTransport("https://example.com", "tok")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(transport.session, "request", return_value=mock_resp) as mock_req:
            result = transport._request("GET", "/api/test")

        mock_req.assert_called_once()
        args, kwargs = mock_req.call_args
        self.assertEqual(args[0], "GET")
        self.assertIn("/api/test", args[1])
        self.assertEqual(result, {"ok": True})

    def test_request_posts_json_body(self):
        """_request sends JSON body for POST/PATCH."""
        transport = HttpTransport("https://example.com", "tok")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(transport.session, "request", return_value=mock_resp) as mock_req:
            transport._request("PATCH", "/api/workers/job-1/status", {"status": "running"})

        _, kwargs = mock_req.call_args
        self.assertEqual(kwargs["json"], {"status": "running"})

    def test_get_job_calls_request(self):
        """get_job uses _request to fetch job data."""
        transport = HttpTransport("https://example.com", "tok")
        with patch.object(transport, "_request", return_value={"job": {"jobId": "j1"}}) as mock:
            result = transport.get_job("j1")
        mock.assert_called_once_with("GET", "/api/jobs/j1")
        self.assertEqual(result, {"jobId": "j1"})

    def test_update_status_calls_request(self):
        """update_status uses _request with PATCH."""
        transport = HttpTransport("https://example.com", "tok")
        with patch.object(transport, "_request") as mock:
            transport.update_status("j1", "running")
        mock.assert_called_once_with("PATCH", "/api/workers/j1/status", {"status": "running"})

    def test_update_status_includes_error(self):
        """update_status includes error field when provided."""
        transport = HttpTransport("https://example.com", "tok")
        with patch.object(transport, "_request") as mock:
            transport.update_status("j1", "failed", error="boom")
        mock.assert_called_once_with(
            "PATCH", "/api/workers/j1/status", {"status": "failed", "error": "boom"},
        )

    def test_download_checkpoint_uses_session_get(self):
        """_download_checkpoint uses session.get with stream=True."""
        transport = HttpTransport("http://localhost:4400", "tok")
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            body = b"model weights data"

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-length": str(len(body))}
            mock_resp.iter_content = MagicMock(return_value=[body])
            mock_resp.raise_for_status = MagicMock()

            with patch.object(transport.session, "get", return_value=mock_resp) as mock_get:
                agent._download_checkpoint({
                    "checkpoint": "experiments/exp-abc/model.pth",
                })

            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            self.assertIn("model.pth", args[0])
            self.assertTrue(kwargs.get("stream"))

            # Verify file was saved
            final_path = os.path.join(tmpdir, "experiments", "exp-abc", "model.pth")
            self.assertTrue(os.path.exists(final_path))
            with open(final_path, "rb") as f:
                self.assertEqual(f.read(), body)

    def test_upload_single_uses_session_put(self):
        """_upload_checkpoint_single uses session.put with file body."""
        transport = HttpTransport("http://localhost:4400", "tok")
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.pth")
            with open(filepath, "wb") as f:
                f.write(b"fake model")

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = '{"ok": true}'

            with patch.object(transport.session, "put", return_value=mock_resp) as mock_put:
                agent._upload_checkpoint_single(filepath, "exp-1", "model.pth", 10)

            mock_put.assert_called_once()
            args, kwargs = mock_put.call_args
            self.assertIn("model.pth", args[0])
            self.assertEqual(kwargs["headers"]["Content-Type"], "application/octet-stream")


# ---------------------------------------------------------------------------
# Log streaming
# ---------------------------------------------------------------------------

class TestLogStreaming(unittest.TestCase):
    """Tests for send_logs on both transport types and log capture in WorkerAgent."""

    def test_redis_transport_send_logs(self):
        """RedisTransport.send_logs pushes to Redis list and publishes."""
        with patch("worker_agent.redis_lib") as mock_redis_lib:
            mock_redis = MagicMock()
            mock_redis_lib.from_url.return_value = mock_redis
            mock_pipe = MagicMock()
            mock_redis.pipeline.return_value = mock_pipe

            transport = RedisTransport("redis://localhost:6379")
            lines = [
                {"timestamp": 1000, "stream": "stdout", "line": "hello"},
                {"timestamp": 1001, "stream": "stderr", "line": "error"},
            ]
            transport.send_logs("job-1", lines)

            # Should RPUSH, LTRIM, PUBLISH, exec
            mock_pipe.rpush.assert_called_once()
            rpush_args = mock_pipe.rpush.call_args[0]
            self.assertEqual(rpush_args[0], "tidal:logs:job-1")

            mock_pipe.ltrim.assert_called_once_with("tidal:logs:job-1", -10000, -1)
            mock_pipe.publish.assert_called_once()
            mock_pipe.execute.assert_called_once()

    def test_http_transport_send_logs(self):
        """HttpTransport.send_logs POSTs to the logs endpoint."""
        transport = HttpTransport("https://example.com", "tok")
        lines = [{"timestamp": 1000, "stream": "stdout", "line": "hello"}]

        with patch.object(transport, "_request") as mock_req:
            transport.send_logs("job-1", lines)

        mock_req.assert_called_once_with(
            "POST", "/api/workers/job-1/logs", {"lines": lines},
        )

    def test_send_logs_resilient_to_failure(self):
        """send_logs should not raise even if transport fails."""
        transport = HttpTransport("https://example.com", "tok")

        with patch.object(transport, "_request", side_effect=Exception("network")):
            # Should not raise
            transport.send_logs("job-1", [{"timestamp": 1, "stream": "stdout", "line": "x"}])

    def test_redis_send_logs_resilient_to_failure(self):
        """RedisTransport.send_logs should not raise on Redis errors."""
        with patch("worker_agent.redis_lib") as mock_redis_lib:
            mock_redis = MagicMock()
            mock_redis_lib.from_url.return_value = mock_redis
            mock_redis.pipeline.side_effect = Exception("Redis down")

            transport = RedisTransport("redis://localhost:6379")
            # Should not raise
            transport.send_logs("job-1", [{"timestamp": 1, "stream": "stdout", "line": "x"}])

    def test_spawn_captures_stdout(self):
        """_spawn_and_monitor captures stdout lines into the log buffer."""
        transport = _make_http_transport()
        agent = _make_agent(transport)
        transport.send_logs = MagicMock()
        transport.read_signal = MagicMock(return_value=None)

        # Run a subprocess that outputs to stdout
        exit_code = agent._spawn_and_monitor(
            ["python", "-c", "print('hello from stdout')"],
            redis_prefix="tidal",
        )

        self.assertEqual(exit_code, 0)

        # send_logs should have been called with the stdout line
        self.assertTrue(transport.send_logs.called, "send_logs should have been called")
        all_lines = []
        for call_args in transport.send_logs.call_args_list:
            all_lines.extend(call_args[0][1])  # lines arg
        stdout_lines = [l for l in all_lines if l["stream"] == "stdout"]
        stdout_text = " ".join(l["line"] for l in stdout_lines)
        self.assertIn("hello from stdout", stdout_text)

    def test_spawn_captures_stderr(self):
        """_spawn_and_monitor captures stderr lines into the log buffer."""
        transport = _make_http_transport()
        agent = _make_agent(transport)
        transport.send_logs = MagicMock()
        transport.read_signal = MagicMock(return_value=None)

        exit_code = agent._spawn_and_monitor(
            ["python", "-c", "import sys; print('error msg', file=sys.stderr)"],
            redis_prefix="tidal",
        )

        self.assertEqual(exit_code, 0)

        all_lines = []
        for call_args in transport.send_logs.call_args_list:
            all_lines.extend(call_args[0][1])
        stderr_lines = [l for l in all_lines if l["stream"] == "stderr"]
        stderr_text = " ".join(l["line"] for l in stderr_lines)
        self.assertIn("error msg", stderr_text)

    def test_spawn_final_flush(self):
        """send_logs is called at least once after the process exits (final flush)."""
        transport = _make_http_transport()
        agent = _make_agent(transport)
        transport.send_logs = MagicMock()
        transport.read_signal = MagicMock(return_value=None)

        agent._spawn_and_monitor(
            ["python", "-c", "print('done')"],
            redis_prefix="tidal",
        )

        # At minimum there should be a final flush call
        self.assertTrue(transport.send_logs.called)


if __name__ == "__main__":
    unittest.main()
