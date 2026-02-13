"""
Worker agent that runs training jobs on behalf of the dashboard orchestrator.

Reads job config from Redis (local) or HTTPS API (remote), spawns the
appropriate training subprocess, monitors for signals (complete/stop),
and reports status back.
"""

import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from urllib.parse import urlparse

import redis as redis_lib


JOBS_HASH = "tidal:jobs"
SIGNAL_PREFIX = "tidal:job:"
HEARTBEAT_PREFIX = "tidal:worker:"
UPDATES_CHANNEL = "tidal:job:updates"

HEARTBEAT_INTERVAL = 10
HEARTBEAT_TTL = 30
SIGNAL_POLL_INTERVAL = 2
UPLOAD_CHUNK_SIZE = 95 * 1024 * 1024  # 95MB — under Cloudflare's 100MB limit
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_READ_SIZE = 1024 * 1024  # 1MB chunks for streaming reads


# ── Transport abstraction ────────────────────────────────────────────

class Transport(ABC):
    @abstractmethod
    def get_job(self, job_id: str) -> dict | None: ...

    @abstractmethod
    def update_status(self, job_id: str, status: str, error: str | None = None) -> None: ...

    @abstractmethod
    def send_heartbeat(self, job_id: str) -> None: ...

    @abstractmethod
    def read_signal(self, job_id: str) -> str | None: ...


class RedisTransport(Transport):
    """Existing behavior — communicates directly with Redis for local jobs."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = redis_lib.from_url(redis_url, decode_responses=True)

    def get_job(self, job_id: str) -> dict | None:
        raw = self.redis.hget(JOBS_HASH, job_id)
        return json.loads(raw) if raw else None

    def update_status(self, job_id: str, status: str, error: str | None = None) -> None:
        try:
            raw = self.redis.hget(JOBS_HASH, job_id)
            if not raw:
                return
            job = json.loads(raw)
            job["status"] = status
            job["updatedAt"] = time.time()
            if error:
                job["error"] = error
            if status in ("completed", "failed"):
                job["completedAt"] = time.time()
            self.redis.hset(JOBS_HASH, job_id, json.dumps(job))
            self.redis.publish(UPDATES_CHANNEL, json.dumps({"jobId": job_id}))
        except Exception as e:
            print(f"Failed to update job status: {e}", file=sys.stderr)

    def send_heartbeat(self, job_id: str) -> None:
        key = f"{HEARTBEAT_PREFIX}{job_id}:heartbeat"
        self.redis.set(key, str(time.time()), ex=HEARTBEAT_TTL)

    def read_signal(self, job_id: str) -> str | None:
        try:
            val = self.redis.get(f"{SIGNAL_PREFIX}{job_id}:signal")
            if val:
                self.redis.delete(f"{SIGNAL_PREFIX}{job_id}:signal")
            return val
        except Exception:
            return None


class HttpTransport(Transport):
    """Communicates with the dashboard Fastify API over HTTPS for remote jobs."""

    def __init__(self, api_url: str, auth_token: str):
        self.api_url = api_url.rstrip("/")
        self.auth_token = auth_token

    def _request(self, method: str, path: str, body: dict | None = None, retries: int = 3) -> dict | None:
        url = f"{self.api_url}{path}"
        last_err = None
        for attempt in range(retries):
            data = json.dumps(body).encode() if body else None
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header("Authorization", f"Bearer {self.auth_token}")
            req.add_header("User-Agent", "TidalWorker/1.0")
            if data:
                req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                body_text = e.read().decode()[:200]
                if e.code in (502, 503, 504) and attempt < retries - 1:
                    wait = 2 ** attempt
                    print(f"HTTP {e.code} on {method} {path} (attempt {attempt + 1}/{retries}) — retrying in {wait}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                print(f"HTTP {e.code} on {method} {path}: {body_text}", file=sys.stderr)
                return None
            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    wait = 2 ** attempt  # 1s, 2s
                    print(f"Request failed {method} {path} (attempt {attempt + 1}/{retries}): {e} — retrying in {wait}s", file=sys.stderr)
                    time.sleep(wait)
        print(f"Request failed {method} {path} after {retries} attempts: {last_err}", file=sys.stderr)
        return None

    def get_job(self, job_id: str) -> dict | None:
        result = self._request("GET", f"/api/jobs/{job_id}")
        return result.get("job") if result else None

    def update_status(self, job_id: str, status: str, error: str | None = None) -> None:
        body: dict = {"status": status}
        if error:
            body["error"] = error
        self._request("PATCH", f"/api/workers/{job_id}/status", body)

    def send_heartbeat(self, job_id: str) -> None:
        self._request("POST", f"/api/workers/{job_id}/heartbeat")

    def read_signal(self, job_id: str) -> str | None:
        result = self._request("GET", f"/api/workers/{job_id}/signal")
        return result.get("signal") if result else None


# ── Worker agent ─────────────────────────────────────────────────────

class WorkerAgent:
    def __init__(self, job_id: str, transport: Transport):
        self.job_id = job_id
        self.transport = transport
        self.process: subprocess.Popen | None = None
        self._heartbeat_stop = threading.Event()
        self._project_root = os.path.dirname(os.path.abspath(__file__))

        # Ensure child process is killed if we receive SIGTERM/SIGINT
        signal.signal(signal.SIGTERM, self._handle_terminate)
        signal.signal(signal.SIGINT, self._handle_terminate)

    def _handle_terminate(self, signum, _frame):
        """Forward termination signal to the training subprocess."""
        sig_name = signal.Signals(signum).name
        print(f"Worker received {sig_name}, terminating child process...", file=sys.stderr)
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._heartbeat_stop.set()
        sys.exit(128 + signum)

    def run(self):
        """Main entry point: read job, start heartbeat, spawn training, monitor."""
        job = self.transport.get_job(self.job_id)
        if not job:
            print(f"Job {self.job_id} not found", file=sys.stderr)
            sys.exit(1)

        job_type = job["config"]["type"]
        config = job["config"]

        # Download checkpoint from dashboard before entering "running" state
        try:
            self._download_checkpoint(config)
        except Exception as e:
            self.transport.update_status(
                self.job_id, "failed", error=str(e)
            )
            return

        self.transport.update_status(self.job_id, "running")
        self._start_heartbeat()

        try:
            if job_type == "lm-training":
                exit_code = self._run_lm_training(config)
            elif job_type == "rl-training":
                exit_code = self._run_rl_training(config)
            else:
                self.transport.update_status(
                    self.job_id, "failed", error=f"Unknown job type: {job_type}"
                )
                return

            if exit_code == 0:
                if isinstance(self.transport, HttpTransport):
                    self._final_checkpoint_sweep()
                self.transport.update_status(self.job_id, "completed")
            else:
                tail = self._get_stderr_tail()
                error_msg = f"Process exited with code {exit_code}"
                if tail:
                    error_msg += f"\n{tail}"
                self.transport.update_status(self.job_id, "failed", error=error_msg)
        except Exception as e:
            self.transport.update_status(self.job_id, "failed", error=str(e))
        finally:
            self._heartbeat_stop.set()

    def _run_lm_training(self, config: dict) -> int:
        args = [
            sys.executable, "Main.py",
            "--config", config["configPath"],
        ]
        if config.get("resumeExpDir"):
            args += ["--resume", config["resumeExpDir"]]
        return self._spawn_and_monitor(args)

    def _run_rl_training(self, config: dict) -> int:
        args = [
            sys.executable, "train_rl.py",
            "--config", config["configPath"],
        ]
        if config.get("rlConfigPath"):
            args += ["--rl-config", config["rlConfigPath"]]
        if config.get("checkpoint"):
            args += ["--checkpoint", config["checkpoint"]]
        if config.get("timesteps"):
            args += ["--timesteps", str(config["timesteps"])]
        return self._spawn_and_monitor(args)

    def _spawn_and_monitor(self, args: list[str]) -> int:
        """Spawn subprocess and poll for signals every 2 seconds."""
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "TIDAL_JOB_ID": self.job_id,
        }

        # Pass API credentials so MetricsLogger can forward metrics remotely
        if hasattr(self.transport, "api_url"):
            env["TIDAL_API_URL"] = self.transport.api_url
            env["TIDAL_AUTH_TOKEN"] = self.transport.auth_token

        # Capture stderr in a ring buffer so we can report crash errors
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()

        self.process = subprocess.Popen(
            args,
            cwd=self._project_root,
            env=env,
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
        )

        # Tee stderr: print to our stderr + keep last 50 lines
        def _read_stderr():
            assert self.process.stderr is not None
            for raw_line in self.process.stderr:
                line = raw_line.decode("utf-8", errors="replace")
                sys.stderr.write(line)
                with self._stderr_lock:
                    self._stderr_lines.append(line.rstrip())
                    if len(self._stderr_lines) > 50:
                        self._stderr_lines.pop(0)

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        while self.process.poll() is None:
            sig = self.transport.read_signal(self.job_id)
            if sig == "complete":
                self._write_complete_signal()
                self.transport.update_status(self.job_id, "completing")
                # Wait for process to finish naturally
                self.process.wait()
                break
            elif sig == "stop":
                self.transport.update_status(self.job_id, "stopping")
                self.process.send_signal(signal.SIGTERM)
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                break

            time.sleep(SIGNAL_POLL_INTERVAL)

        return self.process.returncode or 0

    def _get_stderr_tail(self, lines: int = 10) -> str:
        """Return the last N lines of captured stderr."""
        with self._stderr_lock:
            return "\n".join(self._stderr_lines[-lines:])

    def _write_complete_signal(self):
        """Write sentinel file for Trainer.py to pick up."""
        sentinel = os.path.join(self._project_root, ".training_complete_signal")
        with open(sentinel, "w") as f:
            f.write(str(time.time()))

    def _download_checkpoint(self, config: dict):
        """Download a checkpoint file from the dashboard API before training.

        Guard clauses:
        - Not HttpTransport (local jobs have checkpoints on disk)
        - No 'checkpoint' key in config
        - File already exists on disk
        """
        if not isinstance(self.transport, HttpTransport):
            return
        checkpoint_path = config.get("checkpoint")
        if not checkpoint_path:
            return

        # Parse expId and filename from path containing "experiments/<expId>/<filename>"
        # Handles both relative ("experiments/...") and absolute ("/data/experiments/...")
        parts = checkpoint_path.replace("\\", "/").split("/")
        try:
            idx = parts.index("experiments")
        except ValueError:
            raise RuntimeError(
                f"Cannot parse checkpoint path (no 'experiments' segment): {checkpoint_path}"
            )
        if idx + 2 >= len(parts):
            raise RuntimeError(
                f"Cannot parse checkpoint path: {checkpoint_path}"
            )
        exp_id = parts[idx + 1]
        filename = parts[idx + 2]

        dest = os.path.join(self._project_root, "experiments", exp_id, filename)
        if os.path.exists(dest):
            print(f"Checkpoint already exists: {dest}", file=sys.stderr)
            return

        os.makedirs(os.path.join(self._project_root, "experiments", exp_id), exist_ok=True)

        parsed = urlparse(self.transport.api_url)
        use_https = parsed.scheme == "https"
        host = parsed.hostname
        port = parsed.port or (443 if use_https else 80)
        url_path = (
            f"/api/workers/{self.job_id}/checkpoints/{filename}"
            f"?expId={exp_id}"
        )

        tmp_dest = dest + ".tmp"
        last_err = None

        for attempt in range(DOWNLOAD_MAX_RETRIES):
            try:
                if use_https:
                    conn = http.client.HTTPSConnection(host, port, timeout=300)
                else:
                    conn = http.client.HTTPConnection(host, port, timeout=300)

                conn.putrequest("GET", url_path)
                conn.putheader("Authorization", f"Bearer {self.transport.auth_token}")
                conn.putheader("User-Agent", "TidalWorker/1.0")
                conn.endheaders()

                resp = conn.getresponse()

                if resp.status == 404:
                    self._cleanup_tmp(tmp_dest)
                    raise RuntimeError(
                        f"Checkpoint not found on server (404): {filename}"
                    )

                if resp.status >= 500:
                    last_err = RuntimeError(
                        f"Server error {resp.status} downloading {filename}"
                    )
                    if attempt < DOWNLOAD_MAX_RETRIES - 1:
                        wait = 2 ** attempt
                        print(
                            f"HTTP {resp.status} downloading checkpoint "
                            f"(attempt {attempt + 1}/{DOWNLOAD_MAX_RETRIES}) "
                            f"— retrying in {wait}s",
                            file=sys.stderr,
                        )
                        time.sleep(wait)
                        continue
                    self._cleanup_tmp(tmp_dest)
                    raise RuntimeError(
                        f"Failed to download checkpoint after {DOWNLOAD_MAX_RETRIES} attempts: "
                        f"HTTP {resp.status}"
                    )

                if resp.status != 200:
                    self._cleanup_tmp(tmp_dest)
                    raise RuntimeError(
                        f"Unexpected HTTP {resp.status} downloading {filename}"
                    )

                # Stream response body to temp file
                expected_size = resp.getheader("content-length")
                bytes_written = 0
                with open(tmp_dest, "wb") as f:
                    while True:
                        chunk = resp.read(DOWNLOAD_READ_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        bytes_written += len(chunk)

                conn.close()

                # Verify Content-Length if provided
                if expected_size is not None:
                    expected = int(expected_size)
                    if bytes_written != expected:
                        self._cleanup_tmp(tmp_dest)
                        raise RuntimeError(
                            f"Content-Length mismatch: expected {expected} bytes, "
                            f"got {bytes_written}"
                        )

                # Atomic rename
                os.rename(tmp_dest, dest)
                print(
                    f"Downloaded checkpoint: {filename} ({bytes_written} bytes)",
                    file=sys.stderr,
                )
                return

            except RuntimeError:
                raise
            except Exception as e:
                last_err = e
                if attempt < DOWNLOAD_MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    print(
                        f"Download failed (attempt {attempt + 1}/{DOWNLOAD_MAX_RETRIES}): "
                        f"{e} — retrying in {wait}s",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue

        self._cleanup_tmp(tmp_dest)
        raise RuntimeError(
            f"Failed to download checkpoint after {DOWNLOAD_MAX_RETRIES} attempts: {last_err}"
        )

    @staticmethod
    def _cleanup_tmp(path: str):
        """Remove a temp file if it exists."""
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def _final_checkpoint_sweep(self):
        """Safety net: upload any .pth files that background threads may have missed."""
        exp_base = os.path.join(self._project_root, "experiments")
        if not os.path.isdir(exp_base):
            return

        assert isinstance(self.transport, HttpTransport)
        for exp_id in os.listdir(exp_base):
            exp_dir = os.path.join(exp_base, exp_id)
            if not os.path.isdir(exp_dir):
                continue
            for fname in os.listdir(exp_dir):
                if not fname.endswith(".pth"):
                    continue
                filepath = os.path.join(exp_dir, fname)
                print(f"Final sweep: uploading {filepath}", file=sys.stderr)
                self._upload_checkpoint(filepath, exp_id, fname)

    def _upload_checkpoint(self, filepath: str, exp_id: str, filename: str):
        """Stream a checkpoint file to the dashboard API, chunking if needed."""
        assert isinstance(self.transport, HttpTransport)
        file_size = os.path.getsize(filepath)

        if file_size <= UPLOAD_CHUNK_SIZE:
            self._upload_checkpoint_single(filepath, exp_id, filename, file_size)
        else:
            import math
            total_chunks = math.ceil(file_size / UPLOAD_CHUNK_SIZE)
            print(f"Uploading {filename} ({file_size} bytes) in {total_chunks} chunks", file=sys.stderr)
            for chunk_idx in range(total_chunks):
                offset = chunk_idx * UPLOAD_CHUNK_SIZE
                length = min(UPLOAD_CHUNK_SIZE, file_size - offset)
                self._upload_checkpoint_chunk(
                    filepath, exp_id, filename,
                    chunk_idx, total_chunks, offset, length,
                )

    def _upload_checkpoint_single(
        self, filepath: str, exp_id: str, filename: str, file_size: int,
    ):
        """Upload a complete checkpoint in a single request."""
        assert isinstance(self.transport, HttpTransport)
        parsed = urlparse(self.transport.api_url)
        use_https = parsed.scheme == "https"
        host = parsed.hostname
        port = parsed.port or (443 if use_https else 80)

        url_path = (
            f"/api/workers/{self.job_id}/checkpoints/{filename}"
            f"?expId={exp_id}"
        )

        try:
            if use_https:
                conn = http.client.HTTPSConnection(host, port, timeout=300)
            else:
                conn = http.client.HTTPConnection(host, port, timeout=300)

            conn.putrequest("PUT", url_path)
            conn.putheader("Authorization", f"Bearer {self.transport.auth_token}")
            conn.putheader("Content-Type", "application/octet-stream")
            conn.putheader("Content-Length", str(file_size))
            conn.putheader("User-Agent", "TidalWorker/1.0")
            conn.endheaders()

            with open(filepath, "rb") as f:
                while True:
                    data = f.read(1024 * 1024)
                    if not data:
                        break
                    conn.send(data)

            resp = conn.getresponse()
            body = resp.read().decode()[:200]
            conn.close()

            if 200 <= resp.status < 300:
                print(f"Uploaded {filename} ({file_size} bytes)", file=sys.stderr)
            else:
                print(f"Upload failed for {filename}: HTTP {resp.status} {body}", file=sys.stderr)
        except Exception as e:
            print(f"Upload error for {filename}: {e}", file=sys.stderr)

    def _upload_checkpoint_chunk(
        self, filepath: str, exp_id: str, filename: str,
        chunk_idx: int, total_chunks: int, offset: int, length: int,
    ):
        """Upload one chunk of a checkpoint file."""
        assert isinstance(self.transport, HttpTransport)
        parsed = urlparse(self.transport.api_url)
        use_https = parsed.scheme == "https"
        host = parsed.hostname
        port = parsed.port or (443 if use_https else 80)

        url_path = (
            f"/api/workers/{self.job_id}/checkpoints/{filename}"
            f"?expId={exp_id}&chunk={chunk_idx}&totalChunks={total_chunks}"
        )

        try:
            if use_https:
                conn = http.client.HTTPSConnection(host, port, timeout=300)
            else:
                conn = http.client.HTTPConnection(host, port, timeout=300)

            conn.putrequest("PUT", url_path)
            conn.putheader("Authorization", f"Bearer {self.transport.auth_token}")
            conn.putheader("Content-Type", "application/octet-stream")
            conn.putheader("Content-Length", str(length))
            conn.putheader("User-Agent", "TidalWorker/1.0")
            conn.endheaders()

            sent = 0
            with open(filepath, "rb") as f:
                f.seek(offset)
                while sent < length:
                    read_size = min(1024 * 1024, length - sent)
                    data = f.read(read_size)
                    if not data:
                        break
                    conn.send(data)
                    sent += len(data)

            resp = conn.getresponse()
            body = resp.read().decode()[:200]
            conn.close()

            if 200 <= resp.status < 300:
                print(
                    f"Uploaded chunk {chunk_idx + 1}/{total_chunks} of {filename} ({length} bytes)",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Upload failed for chunk {chunk_idx + 1}/{total_chunks} of {filename}: HTTP {resp.status} {body}",
                    file=sys.stderr,
                )
        except Exception as e:
            print(
                f"Upload error for chunk {chunk_idx + 1}/{total_chunks} of {filename}: {e}",
                file=sys.stderr,
            )

    def _start_heartbeat(self):
        """Thread: send heartbeat every 10s."""
        def _beat():
            while not self._heartbeat_stop.wait(HEARTBEAT_INTERVAL):
                try:
                    self.transport.send_heartbeat(self.job_id)
                except Exception:
                    pass

        t = threading.Thread(target=_beat, daemon=True)
        t.start()


def main():
    parser = argparse.ArgumentParser(description="Tidal training worker agent")
    parser.add_argument("--job-id", required=True, help="Job ID to execute")
    parser.add_argument(
        "--redis-url",
        default=None,
        help="Redis connection URL (for local mode)",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="Dashboard API URL (for remote mode, e.g. https://ai.michelleeby.com)",
    )
    parser.add_argument(
        "--auth-token",
        default=None,
        help="Auth token for dashboard API (remote mode)",
    )
    args = parser.parse_args()

    if args.api_url:
        if not args.auth_token:
            print("--auth-token is required when using --api-url", file=sys.stderr)
            sys.exit(1)
        transport = HttpTransport(args.api_url, args.auth_token)
    else:
        redis_url = args.redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        transport = RedisTransport(redis_url)

    agent = WorkerAgent(args.job_id, transport)
    agent.run()


if __name__ == "__main__":
    main()
