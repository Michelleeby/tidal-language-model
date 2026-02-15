"""
Worker agent that runs training jobs on behalf of the dashboard orchestrator.

Reads job config from Redis (local) or HTTPS API (remote), spawns the
appropriate training subprocess, monitors for signals (complete/stop),
and reports status back.  Streams stdout/stderr logs to the dashboard.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod

import requests as requests_lib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ruamel.yaml import YAML

import redis as redis_lib


JOBS_HASH = "tidal:jobs"
SIGNAL_PREFIX = "tidal:job:"
HEARTBEAT_PREFIX = "tidal:worker:"
UPDATES_CHANNEL = "tidal:job:updates"

_ENV_ALLOWLIST_EXACT = frozenset({
    "PATH", "HOME", "USER", "SHELL",
    "LANG", "LC_ALL", "LC_CTYPE", "TERM",
    "PYTHONPATH", "VIRTUAL_ENV", "PYTHONHASHSEED",
    "LD_LIBRARY_PATH", "LIBRARY_PATH",
    "HF_HOME", "HF_DATASETS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME",
    "REDIS_URL", "METRICS_REDIS_PREFIX",
    "TMPDIR", "TEMP", "TMP",
})
_ENV_ALLOWLIST_PREFIXES = (
    "CUDA_", "NVIDIA_", "NCCL_",
    "TRAINING_", "TIDAL_",
    "TORCH_",
)

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

    @abstractmethod
    def send_logs(self, job_id: str, lines: list[dict]) -> None: ...


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

    def send_logs(self, job_id: str, lines: list[dict]) -> None:
        try:
            key = f"tidal:logs:{job_id}"
            pipe = self.redis.pipeline()
            serialized = [json.dumps(l) for l in lines]
            pipe.rpush(key, *serialized)
            pipe.ltrim(key, -10000, -1)
            pipe.publish("tidal:logs:stream", json.dumps({"jobId": job_id, "lines": lines}))
            pipe.execute()
        except Exception as e:
            print(f"Failed to send logs: {e}", file=sys.stderr)


class HttpTransport(Transport):
    """Communicates with the dashboard Fastify API over HTTPS for remote jobs."""

    def __init__(self, api_url: str, auth_token: str):
        self.api_url = api_url.rstrip("/")
        self.auth_token = auth_token

        # Build a session with default auth headers and retry strategy
        self.session = requests_lib.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {auth_token}",
            "User-Agent": "TidalWorker/1.0",
        })
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _request(self, method: str, path: str, body: dict | None = None) -> dict | None:
        url = f"{self.api_url}{path}"
        try:
            resp = self.session.request(method, url, json=body, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests_lib.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text[:200] if e.response is not None else ""
            print(f"HTTP {status} on {method} {path}: {text}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Request failed {method} {path}: {e}", file=sys.stderr)
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

    def send_logs(self, job_id: str, lines: list[dict]) -> None:
        try:
            self._request("POST", f"/api/workers/{job_id}/logs", {"lines": lines})
        except Exception as e:
            print(f"Failed to send logs: {e}", file=sys.stderr)


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

    # ── Manifest helpers ────────────────────────────────────────────────

    def _load_manifest(self, plugin_name: str) -> dict:
        """Load a plugin manifest from plugins/<name>/manifest.yaml."""
        plugin_dir = os.path.join(self._project_root, "plugins", plugin_name)
        return self._load_manifest_from_dir(plugin_dir)

    def _load_manifest_from_dir(self, plugin_dir: str) -> dict:
        """Load a plugin manifest from an arbitrary directory."""
        manifest_path = os.path.join(plugin_dir, "manifest.yaml")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Plugin manifest not found: {manifest_path}"
            )
        yaml = YAML()
        with open(manifest_path) as f:
            return dict(yaml.load(f))

    def _find_phase(self, manifest: dict, job_type: str) -> dict:
        """Look up a training phase by ID in a manifest."""
        for phase in manifest["trainingPhases"]:
            if phase["id"] == job_type:
                return dict(phase)
        available = [p["id"] for p in manifest["trainingPhases"]]
        raise ValueError(
            f"Unknown job type '{job_type}' for plugin '{manifest['name']}'. "
            f"Available phases: {available}"
        )

    def _build_command(self, phase: dict, config: dict, plugin_dir: str) -> list[str]:
        """Build subprocess args from a manifest phase and job config.

        For system plugins (under plugins/), uses the full dotted path so
        ``python -m plugins.tidal.Main`` works with CWD on sys.path.

        For user plugins (under user-plugins/ or other non-plugins/ dirs),
        uses just ``<basename>.Main`` — the caller must set PYTHONPATH to
        include the parent directory.
        """
        rel_dir = os.path.relpath(plugin_dir, self._project_root)
        module_name = phase["entrypoint"].removesuffix(".py")

        if rel_dir.startswith("plugins" + os.sep) or rel_dir == "plugins":
            # System plugin: plugins.tidal.Main
            module_path = rel_dir.replace(os.sep, ".") + "." + module_name
        else:
            # User plugin: my_model.Main (PYTHONPATH-based resolution)
            basename = os.path.basename(plugin_dir)
            module_path = basename + "." + module_name

        args = [sys.executable, "-m", module_path]

        # Map from manifest arg key to config field name
        ARG_CONFIG_MAP = {
            "config": "configPath",
            "resume": "resumeExpDir",
            "rlConfig": "rlConfigPath",
            "checkpoint": "checkpoint",
            "timesteps": "timesteps",
        }

        for arg_key, cli_flag in phase.get("args", {}).items():
            config_key = ARG_CONFIG_MAP.get(arg_key, arg_key)
            value = config.get(config_key)
            if value is not None:
                args += [cli_flag, str(value)]

        return args

    # ── Main entry point ──────────────────────────────────────────────

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
            exit_code = self._run_training(config)

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

    def _run_training(self, config: dict) -> int:
        """Build and run a training command from the plugin manifest.

        Supports two modes:
        1. pluginDir: use the directory directly, set PYTHONPATH so the
           module can be found (local user plugin path).
        2. Default: standard plugin in plugins/<name>/ — works for both
           system plugins and remote user plugins (whose repo is cloned
           into plugins/<name>/ by the onstart script before the worker
           starts).
        """
        plugin_name = config.get("plugin", "tidal")
        extra_env = None

        if config.get("pluginDir"):
            # Local user plugin: resolve relative to project root
            plugin_dir = os.path.join(self._project_root, config["pluginDir"])
            # PYTHONPATH includes parent so `python -m <name>.Main` works
            parent = os.path.dirname(plugin_dir)
            extra_env = {"PYTHONPATH": parent}
        else:
            # System plugin or remote user plugin (already at plugins/<name>/)
            plugin_dir = os.path.join(self._project_root, "plugins", plugin_name)

        manifest = self._load_manifest_from_dir(plugin_dir)
        phase = self._find_phase(manifest, config["type"])
        args = self._build_command(phase, config, plugin_dir)
        redis_prefix = manifest.get("metrics", {}).get("redisPrefix", "tidal")
        return self._spawn_and_monitor(args, redis_prefix=redis_prefix, extra_env=extra_env)

    def _spawn_and_monitor(
        self,
        args: list[str],
        redis_prefix: str = "tidal",
        extra_env: dict[str, str] | None = None,
    ) -> int:
        """Spawn subprocess and poll for signals every 2 seconds."""
        # Build filtered env from allowlist instead of copying everything
        env = {}
        for key, value in os.environ.items():
            if key in _ENV_ALLOWLIST_EXACT or any(
                key.startswith(p) for p in _ENV_ALLOWLIST_PREFIXES
            ):
                env[key] = value

        # Merge extra_env (e.g. PYTHONPATH for user plugins)
        if extra_env:
            env.update(extra_env)

        # Job-specific variables
        env["PYTHONUNBUFFERED"] = "1"
        env["TRAINING_JOB_ID"] = self.job_id
        # Keep legacy name for backward compatibility during transition
        env["TIDAL_JOB_ID"] = self.job_id
        env["METRICS_REDIS_PREFIX"] = redis_prefix

        # Pass API credentials so MetricsLogger can forward metrics remotely
        if hasattr(self.transport, "api_url"):
            env["TRAINING_API_URL"] = self.transport.api_url
            env["TRAINING_AUTH_TOKEN"] = self.transport.auth_token
            # Keep legacy names for backward compatibility during transition
            env["TIDAL_API_URL"] = self.transport.api_url
            env["TIDAL_AUTH_TOKEN"] = self.transport.auth_token

        # Capture stderr in a ring buffer so we can report crash errors
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()

        # Log buffer for streaming to dashboard
        self._log_buffer: list[dict] = []
        self._log_lock = threading.Lock()

        self.process = subprocess.Popen(
            args,
            cwd=self._project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Tee stdout: print to our stdout + buffer log entries
        def _read_stdout():
            assert self.process.stdout is not None
            for raw_line in self.process.stdout:
                line = raw_line.decode("utf-8", errors="replace")
                sys.stdout.write(line)
                sys.stdout.flush()
                with self._log_lock:
                    self._log_buffer.append({
                        "timestamp": time.time(),
                        "stream": "stdout",
                        "line": line.rstrip(),
                    })

        # Tee stderr: print to our stderr + keep last 50 lines + buffer log entries
        def _read_stderr():
            assert self.process.stderr is not None
            for raw_line in self.process.stderr:
                line = raw_line.decode("utf-8", errors="replace")
                sys.stderr.write(line)
                with self._stderr_lock:
                    self._stderr_lines.append(line.rstrip())
                    if len(self._stderr_lines) > 50:
                        self._stderr_lines.pop(0)
                with self._log_lock:
                    self._log_buffer.append({
                        "timestamp": time.time(),
                        "stream": "stderr",
                        "line": line.rstrip(),
                    })

        stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stdout_thread.start()
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

            # Flush log buffer every poll cycle
            self._flush_logs()

            time.sleep(SIGNAL_POLL_INTERVAL)

        # Wait for reader threads to drain
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        # Final flush of any remaining log lines
        self._flush_logs()

        return self.process.returncode or 0

    def _flush_logs(self):
        """Send buffered log lines to the transport."""
        with self._log_lock:
            if not self._log_buffer:
                return
            batch = self._log_buffer[:]
            self._log_buffer.clear()
        try:
            self.transport.send_logs(self.job_id, batch)
        except Exception as e:
            print(f"Failed to flush logs: {e}", file=sys.stderr)

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

        local_path = os.path.join("experiments", exp_id, filename)
        dest = os.path.join(self._project_root, local_path)
        if os.path.exists(dest):
            print(f"Checkpoint already exists: {dest}", file=sys.stderr)
            config["checkpoint"] = local_path
            return

        os.makedirs(os.path.join(self._project_root, "experiments", exp_id), exist_ok=True)

        url = (
            f"{self.transport.api_url}"
            f"/api/workers/{self.job_id}/checkpoints/{filename}"
            f"?expId={exp_id}"
        )

        tmp_dest = dest + ".tmp"
        last_err = None

        for attempt in range(DOWNLOAD_MAX_RETRIES):
            try:
                resp = self.transport.session.get(url, stream=True, timeout=300)

                if resp.status_code == 404:
                    self._cleanup_tmp(tmp_dest)
                    raise RuntimeError(
                        f"Checkpoint not found on server (404): {filename}"
                    )

                if resp.status_code >= 500:
                    last_err = RuntimeError(
                        f"Server error {resp.status_code} downloading {filename}"
                    )
                    if attempt < DOWNLOAD_MAX_RETRIES - 1:
                        wait = 2 ** attempt
                        print(
                            f"HTTP {resp.status_code} downloading checkpoint "
                            f"(attempt {attempt + 1}/{DOWNLOAD_MAX_RETRIES}) "
                            f"— retrying in {wait}s",
                            file=sys.stderr,
                        )
                        time.sleep(wait)
                        continue
                    self._cleanup_tmp(tmp_dest)
                    raise RuntimeError(
                        f"Failed to download checkpoint after {DOWNLOAD_MAX_RETRIES} attempts: "
                        f"HTTP {resp.status_code}"
                    )

                if resp.status_code != 200:
                    self._cleanup_tmp(tmp_dest)
                    raise RuntimeError(
                        f"Unexpected HTTP {resp.status_code} downloading {filename}"
                    )

                # Stream response body to temp file
                expected_size = resp.headers.get("content-length")
                bytes_written = 0
                with open(tmp_dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=DOWNLOAD_READ_SIZE):
                        if chunk:
                            f.write(chunk)
                            bytes_written += len(chunk)

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
                # Rewrite config to local relative path so train_rl.py finds it
                config["checkpoint"] = local_path
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
        """Safety net: upload .pth and .json result files that need to reach the dashboard."""
        exp_base = os.path.join(self._project_root, "experiments")
        if not os.path.isdir(exp_base):
            return

        assert isinstance(self.transport, HttpTransport)
        for exp_id in os.listdir(exp_base):
            exp_dir = os.path.join(exp_base, exp_id)
            if not os.path.isdir(exp_dir):
                continue
            for fname in os.listdir(exp_dir):
                if not (fname.endswith(".pth") or fname.endswith(".json")):
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
        url = (
            f"{self.transport.api_url}"
            f"/api/workers/{self.job_id}/checkpoints/{filename}"
            f"?expId={exp_id}"
        )

        try:
            with open(filepath, "rb") as f:
                resp = self.transport.session.put(
                    url,
                    data=f,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(file_size),
                    },
                    timeout=300,
                )

            if 200 <= resp.status_code < 300:
                print(f"Uploaded {filename} ({file_size} bytes)", file=sys.stderr)
            else:
                print(f"Upload failed for {filename}: HTTP {resp.status_code} {resp.text[:200]}", file=sys.stderr)
        except Exception as e:
            print(f"Upload error for {filename}: {e}", file=sys.stderr)

    def _upload_checkpoint_chunk(
        self, filepath: str, exp_id: str, filename: str,
        chunk_idx: int, total_chunks: int, offset: int, length: int,
    ):
        """Upload one chunk of a checkpoint file."""
        assert isinstance(self.transport, HttpTransport)
        url = (
            f"{self.transport.api_url}"
            f"/api/workers/{self.job_id}/checkpoints/{filename}"
            f"?expId={exp_id}&chunk={chunk_idx}&totalChunks={total_chunks}"
        )

        try:
            with open(filepath, "rb") as f:
                f.seek(offset)
                data = f.read(length)

            resp = self.transport.session.put(
                url,
                data=data,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(length),
                },
                timeout=300,
            )

            if 200 <= resp.status_code < 300:
                print(
                    f"Uploaded chunk {chunk_idx + 1}/{total_chunks} of {filename} ({length} bytes)",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Upload failed for chunk {chunk_idx + 1}/{total_chunks} of {filename}: HTTP {resp.status_code} {resp.text[:200]}",
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
                except Exception as e:
                    print(f"Heartbeat failed for job {self.job_id}: {e}", file=sys.stderr)

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
