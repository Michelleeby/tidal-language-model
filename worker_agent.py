"""
Worker agent that runs training jobs on behalf of the dashboard orchestrator.

Reads job config from Redis (local) or HTTPS API (remote), spawns the
appropriate training subprocess, monitors for signals (complete/stop),
and reports status back.
"""

import argparse
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

import redis as redis_lib


JOBS_HASH = "tidal:jobs"
SIGNAL_PREFIX = "tidal:job:"
HEARTBEAT_PREFIX = "tidal:worker:"
UPDATES_CHANNEL = "tidal:job:updates"

HEARTBEAT_INTERVAL = 10
HEARTBEAT_TTL = 30
SIGNAL_POLL_INTERVAL = 2


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

    def _request(self, method: str, path: str, body: dict | None = None) -> dict | None:
        url = f"{self.api_url}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.auth_token}")
        req.add_header("User-Agent", "TidalWorker/1.0")
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code} on {method} {path}: {e.read().decode()}", file=sys.stderr)
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
                self.transport.update_status(self.job_id, "completed")
            else:
                self.transport.update_status(
                    self.job_id, "failed", error=f"Process exited with code {exit_code}"
                )
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
        return self._spawn_and_monitor(args)

    def _spawn_and_monitor(self, args: list[str]) -> int:
        """Spawn subprocess and poll for signals every 2 seconds."""
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "TIDAL_JOB_ID": self.job_id,
        }

        self.process = subprocess.Popen(
            args,
            cwd=self._project_root,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

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

    def _write_complete_signal(self):
        """Write sentinel file for Trainer.py to pick up."""
        sentinel = os.path.join(self._project_root, ".training_complete_signal")
        with open(sentinel, "w") as f:
            f.write(str(time.time()))

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
