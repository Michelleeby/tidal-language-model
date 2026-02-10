import json
import os
import time
import logging
import threading
from collections import deque


logger = logging.getLogger(__name__)


class MetricsLogger:
    """
    Training metrics logger with Redis pub/sub and JSONL fallback.

    Writes metrics to both Redis (for real-time dashboard streaming) and
    JSONL on disk (for archival and offline access). When Redis is unavailable,
    falls back to disk-only mode with a single warning.
    """

    def __init__(self, experiment_dir: str, max_history: int = 10000):
        self.experiment_dir = experiment_dir
        self.max_history = max_history
        self.exp_id = os.path.basename(experiment_dir)

        self.metrics_dir = os.path.join(experiment_dir, "dashboard_metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.metrics_history = deque(maxlen=max_history)

        self.metrics_file = os.path.join(self.metrics_dir, "metrics.jsonl")
        self.latest_file = os.path.join(self.metrics_dir, "latest.json")
        self.status_file = os.path.join(self.metrics_dir, "status.json")

        self.lock = threading.Lock()

        self._redis = None
        self._redis_available = True
        self._redis_warned = False
        self._init_redis()

        self._initialize_files()

    def _init_redis(self):
        """Try to connect to Redis. Gracefully degrade if unavailable."""
        try:
            import redis
            url = os.environ.get("REDIS_URL", "redis://localhost:6379")
            self._redis = redis.from_url(url, decode_responses=True)
            self._redis.ping()
        except Exception:
            self._redis = None
            self._redis_available = False
            if not self._redis_warned:
                logger.warning("Redis unavailable — metrics will be written to disk only")
                self._redis_warned = True

    def _redis_write(self, callback):
        """Execute a Redis write, disabling Redis on failure."""
        if not self._redis_available or self._redis is None:
            return
        try:
            callback(self._redis)
        except Exception:
            self._redis_available = False
            if not self._redis_warned:
                logger.warning("Redis connection lost — falling back to disk only")
                self._redis_warned = True

    def _initialize_files(self):
        open(self.metrics_file, "w").close()
        self._update_status({
            "status": "initialized",
            "start_time": time.time(),
            "last_update": time.time(),
        })

    def _update_status(self, status_data: dict):
        with open(self.status_file, "w") as f:
            json.dump(status_data, f, indent=2)

        def _write(r):
            r.set(
                f"tidal:status:{self.exp_id}",
                json.dumps(status_data),
                ex=900,
            )
            r.sadd("tidal:experiments", self.exp_id)
        self._redis_write(_write)

    def log_metrics(self, metrics: dict, step: int):
        """Log metrics for a training step."""
        with self.lock:
            data_point = {
                "step": step,
                "timestamp": time.time(),
                **metrics,
            }

            self.metrics_history.append(data_point)

            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(data_point) + "\n")

            with open(self.latest_file, "w") as f:
                json.dump(data_point, f, indent=2)

            self._update_status({
                "status": "training",
                "current_step": step,
                "last_update": time.time(),
                "total_metrics_logged": len(self.metrics_history),
            })

            serialized = json.dumps(data_point)

            def _write(r):
                pipe = r.pipeline()
                pipe.set(f"tidal:metrics:{self.exp_id}:latest", serialized, ex=600)
                pipe.rpush(f"tidal:metrics:{self.exp_id}:history", serialized)
                pipe.ltrim(f"tidal:metrics:{self.exp_id}:history", -50000, -1)
                pipe.execute()
            self._redis_write(_write)

    def log_rl_metrics(self, rl_history: dict, global_step: int):
        """Log RL training metrics."""
        with self.lock:
            data_point = {
                "global_step": global_step,
                "timestamp": time.time(),
                "history": rl_history,
            }

            rl_metrics_file = os.path.join(self.metrics_dir, "rl_metrics.jsonl")
            with open(rl_metrics_file, "a") as f:
                f.write(json.dumps(data_point) + "\n")

            serialized = json.dumps(data_point)

            def _write(r):
                r.set(f"tidal:rl:{self.exp_id}:latest", serialized, ex=600)
            self._redis_write(_write)

    def finalize(self):
        """Mark training as complete."""
        with self.lock:
            self._update_status({
                "status": "completed",
                "end_time": time.time(),
                "total_steps": self.metrics_history[-1]["step"] if self.metrics_history else 0,
            })

    def get_metrics_history(self, last_n: int = None) -> list:
        with self.lock:
            if last_n:
                return list(self.metrics_history)[-last_n:]
            return list(self.metrics_history)
