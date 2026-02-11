"""
Worker agent that runs training jobs on behalf of the dashboard orchestrator.

Reads job config from Redis, spawns the appropriate training subprocess,
monitors for signals (complete/stop), and reports status back via Redis.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

import redis as redis_lib


JOBS_HASH = "tidal:jobs"
SIGNAL_PREFIX = "tidal:job:"
HEARTBEAT_PREFIX = "tidal:worker:"
UPDATES_CHANNEL = "tidal:job:updates"

HEARTBEAT_INTERVAL = 10
HEARTBEAT_TTL = 30
SIGNAL_POLL_INTERVAL = 2


class WorkerAgent:
    def __init__(self, job_id: str, redis_url: str):
        self.job_id = job_id
        self.redis_url = redis_url
        self.redis = redis_lib.from_url(redis_url, decode_responses=True)
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
        job = self._get_job()
        if not job:
            print(f"Job {self.job_id} not found in Redis", file=sys.stderr)
            sys.exit(1)

        job_type = job["config"]["type"]
        config = job["config"]

        self._update_status("running")
        self._start_heartbeat()

        try:
            if job_type == "lm-training":
                exit_code = self._run_lm_training(config)
            elif job_type == "rl-training":
                exit_code = self._run_rl_training(config)
            else:
                self._update_status("failed", error=f"Unknown job type: {job_type}")
                return

            if exit_code == 0:
                self._update_status("completed")
            else:
                self._update_status("failed", error=f"Process exited with code {exit_code}")
        except Exception as e:
            self._update_status("failed", error=str(e))
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
            "REDIS_URL": self.redis_url,
        }

        self.process = subprocess.Popen(
            args,
            cwd=self._project_root,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        while self.process.poll() is None:
            sig = self._read_signal()
            if sig == "complete":
                self._write_complete_signal()
                self._clear_signal()
                self._update_status("completing")
                # Wait for process to finish naturally
                self.process.wait()
                break
            elif sig == "stop":
                self._clear_signal()
                self._update_status("stopping")
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
        """Thread: SET worker heartbeat key every 10s, TTL 30s."""
        def _beat():
            key = f"{HEARTBEAT_PREFIX}{self.job_id}:heartbeat"
            while not self._heartbeat_stop.wait(HEARTBEAT_INTERVAL):
                try:
                    self.redis.set(key, str(time.time()), ex=HEARTBEAT_TTL)
                except Exception:
                    pass

        t = threading.Thread(target=_beat, daemon=True)
        t.start()

    def _update_status(self, status: str, error: str | None = None):
        """Update job in Redis hash + PUBLISH on updates channel."""
        try:
            raw = self.redis.hget(JOBS_HASH, self.job_id)
            if not raw:
                return
            job = json.loads(raw)
            job["status"] = status
            job["updatedAt"] = time.time()
            if error:
                job["error"] = error
            if status == "completed" or status == "failed":
                job["completedAt"] = time.time()
            self.redis.hset(JOBS_HASH, self.job_id, json.dumps(job))
            self.redis.publish(UPDATES_CHANNEL, json.dumps({"jobId": self.job_id}))
        except Exception as e:
            print(f"Failed to update job status: {e}", file=sys.stderr)

    def _get_job(self) -> dict | None:
        raw = self.redis.hget(JOBS_HASH, self.job_id)
        return json.loads(raw) if raw else None

    def _read_signal(self) -> str | None:
        try:
            return self.redis.get(f"{SIGNAL_PREFIX}{self.job_id}:signal")
        except Exception:
            return None

    def _clear_signal(self):
        try:
            self.redis.delete(f"{SIGNAL_PREFIX}{self.job_id}:signal")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Tidal training worker agent")
    parser.add_argument("--job-id", required=True, help="Job ID to execute")
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("REDIS_URL", "redis://localhost:6379"),
        help="Redis connection URL",
    )
    args = parser.parse_args()

    agent = WorkerAgent(args.job_id, args.redis_url)
    agent.run()


if __name__ == "__main__":
    main()
