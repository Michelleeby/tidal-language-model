"""Tests for WorkerAgent checkpoint download functionality."""

import http.client
import json
import os
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call

from worker_agent import WorkerAgent, RedisTransport, HttpTransport


class FakeResponse:
    """Minimal http.client response stub for _download_checkpoint tests."""

    def __init__(self, status, body=b"", headers=None):
        self.status = status
        self._body = body
        self._headers = headers or {}

    def getheader(self, name, default=None):
        return self._headers.get(name, default)

    def read(self, amt=None):
        if amt is None:
            chunk = self._body
            self._body = b""
            return chunk
        chunk = self._body[:amt]
        self._body = self._body[amt:]
        return chunk

    def close(self):
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
            with patch("http.client.HTTPConnection") as mock_conn_cls:
                agent._download_checkpoint({
                    "checkpoint": "experiments/exp-abc/model.pth",
                })
                mock_conn_cls.assert_not_called()


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

            fake_resp = FakeResponse(200, body, {
                "content-length": str(len(body)),
            })
            mock_conn = MagicMock()
            mock_conn.getresponse.return_value = fake_resp

            with patch("http.client.HTTPConnection", return_value=mock_conn):
                agent._download_checkpoint({
                    "checkpoint": "experiments/abc-123/transformer-lm_v1.0.0.pth",
                })

            # Verify the request path includes expId and filename
            mock_conn.putrequest.assert_called_once()
            req_path = mock_conn.putrequest.call_args[0][1]
            self.assertIn("transformer-lm_v1.0.0.pth", req_path)
            self.assertIn("expId=abc-123", req_path)

    def test_creates_experiment_directory(self):
        """Creates experiments/<expId>/ if it doesn't exist."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir
            body = b"model data"

            fake_resp = FakeResponse(200, body, {
                "content-length": str(len(body)),
            })
            mock_conn = MagicMock()
            mock_conn.getresponse.return_value = fake_resp

            with patch("http.client.HTTPConnection", return_value=mock_conn):
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

            fake_resp = FakeResponse(200, body, {
                "content-length": str(len(body)),
            })
            mock_conn = MagicMock()
            mock_conn.getresponse.return_value = fake_resp

            with patch("http.client.HTTPConnection", return_value=mock_conn):
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
            fake_resp = FakeResponse(200, b"x" * 500, {
                "content-length": "1000",
            })
            mock_conn = MagicMock()
            mock_conn.getresponse.return_value = fake_resp

            with patch("http.client.HTTPConnection", return_value=mock_conn):
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

            fail_resp = FakeResponse(502, b"Bad Gateway")
            ok_resp = FakeResponse(200, body, {
                "content-length": str(len(body)),
            })

            mock_conn = MagicMock()
            mock_conn.getresponse.side_effect = [fail_resp, fail_resp, ok_resp]

            with patch("http.client.HTTPConnection", return_value=mock_conn):
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

            fake_resp = FakeResponse(404, b"Not found")
            mock_conn = MagicMock()
            mock_conn.getresponse.return_value = fake_resp

            with patch("http.client.HTTPConnection", return_value=mock_conn):
                with self.assertRaises(RuntimeError) as ctx:
                    agent._download_checkpoint({
                        "checkpoint": "experiments/exp-1/model.pth",
                    })
                self.assertIn("404", str(ctx.exception))

            # Should NOT have retried
            self.assertEqual(mock_conn.getresponse.call_count, 1)

    @patch("time.sleep")
    def test_raises_after_all_retries_exhausted(self, mock_sleep):
        """Raises RuntimeError after 3 consecutive 5xx failures."""
        transport = _make_http_transport()
        agent = _make_agent(transport)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent._project_root = tmpdir

            fail_resp_1 = FakeResponse(503, b"Unavailable")
            fail_resp_2 = FakeResponse(503, b"Unavailable")
            fail_resp_3 = FakeResponse(503, b"Unavailable")

            mock_conn = MagicMock()
            mock_conn.getresponse.side_effect = [fail_resp_1, fail_resp_2, fail_resp_3]

            with patch("http.client.HTTPConnection", return_value=mock_conn):
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

            ok_resp = FakeResponse(200, body, {
                "content-length": str(len(body)),
            })

            mock_conn = MagicMock()
            mock_conn.getresponse.side_effect = [
                ConnectionError("refused"),
                ok_resp,
            ]

            with patch("http.client.HTTPConnection", return_value=mock_conn):
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

            fake_resp = FakeResponse(404, b"Not found")
            mock_conn = MagicMock()
            mock_conn.getresponse.return_value = fake_resp

            with patch("http.client.HTTPConnection", return_value=mock_conn):
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
            with patch.object(agent, "_run_rl_training", return_value=0):
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
            with patch.object(agent, "_run_rl_training") as mock_train:
                agent.run()
                mock_train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
