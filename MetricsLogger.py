import json
import math
import os
import time
import logging
import threading
from collections import deque

import requests as requests_lib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)

HTTP_BATCH_SIZE = 50
HTTP_FLUSH_INTERVAL = 30  # seconds
UPLOAD_CHUNK_SIZE = 95 * 1024 * 1024  # 95MB — under Cloudflare's 100MB limit


class MetricsLogger:
    """
    Training metrics logger with Redis pub/sub and JSONL fallback.

    Writes metrics to both Redis (for real-time dashboard streaming) and
    JSONL on disk (for archival and offline access). When Redis is unavailable,
    falls back to disk-only mode with a single warning.
    """

    def __init__(self, experiment_dir: str, max_history: int = 10000, reset_metrics: bool = True, redis_prefix: str | None = None):
        self.experiment_dir = experiment_dir
        self.max_history = max_history
        self.exp_id = os.path.basename(experiment_dir)
        self._redis_prefix = redis_prefix or os.environ.get("METRICS_REDIS_PREFIX", "tidal")

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

        # HTTP forwarding for remote workers (vast.ai → dashboard API)
        # Accept both TRAINING_* (new) and TIDAL_* (legacy) env vars
        self._http_url = os.environ.get("TRAINING_API_URL") or os.environ.get("TIDAL_API_URL")
        self._http_token = os.environ.get("TRAINING_AUTH_TOKEN") or os.environ.get("TIDAL_AUTH_TOKEN")
        self._http_job_id = os.environ.get("TRAINING_JOB_ID") or os.environ.get("TIDAL_JOB_ID")
        self._http_enabled = bool(
            self._http_url and self._http_token and self._http_job_id
        )
        self._http_batch: list[dict] = []
        self._http_last_status: dict | None = None
        self._http_flush_timer: threading.Timer | None = None
        self._http_session: requests_lib.Session | None = None
        if self._http_enabled:
            self._http_url = self._http_url.rstrip("/")
            self._http_session = requests_lib.Session()
            self._http_session.headers.update({
                "Authorization": f"Bearer {self._http_token}",
                "User-Agent": "TidalMetrics/1.0",
            })
            retry = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry)
            self._http_session.mount("http://", adapter)
            self._http_session.mount("https://", adapter)

            logger.info("HTTP metrics forwarding enabled → %s", self._http_url)
            self._schedule_http_flush()

        self._initialize_files(reset_metrics=reset_metrics)

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

    # ── HTTP forwarding ────────────────────────────────────────────────

    def _schedule_http_flush(self):
        """Schedule the next periodic flush."""
        self._http_flush_timer = threading.Timer(HTTP_FLUSH_INTERVAL, self._periodic_flush)
        self._http_flush_timer.daemon = True
        self._http_flush_timer.start()

    def _periodic_flush(self):
        """Timer callback: flush batch and reschedule."""
        self._flush_http_batch()
        self._schedule_http_flush()

    def _enqueue_http(self, point: dict):
        """Add a point to the HTTP batch, flushing if batch is full."""
        if not self._http_enabled:
            return
        self._http_batch.append(point)
        if len(self._http_batch) >= HTTP_BATCH_SIZE:
            self._flush_http_batch()

    def _flush_http_batch(self):
        """Send buffered points to the dashboard API."""
        if not self._http_enabled:
            return
        points = self._http_batch
        status = self._http_last_status
        if not points and not status:
            return
        self._http_batch = []

        body: dict = {"expId": self.exp_id}
        if points:
            body["points"] = points
        if status:
            body["status"] = status

        # Fire in background thread to avoid blocking training
        threading.Thread(
            target=self._http_forward,
            args=(body,),
            daemon=True,
        ).start()

    def _http_forward(self, body: dict):
        """POST metrics to the dashboard API."""
        url = f"{self._http_url}/api/workers/{self._http_job_id}/metrics"
        try:
            resp = self._http_session.post(url, json=body, timeout=30)
            resp.raise_for_status()
        except requests_lib.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            text = e.response.text[:200] if e.response is not None else ""
            logger.warning("HTTP %s forwarding metrics: %s", status, text)
        except Exception as e:
            logger.warning("Metrics POST failed: %s", e)

    def _enqueue_rl_http(self, data_point: dict):
        """Forward RL metrics to the dashboard API via the rlLatest field."""
        if not self._http_enabled:
            return
        body: dict = {"expId": self.exp_id, "rlLatest": data_point}
        threading.Thread(
            target=self._http_forward,
            args=(body,),
            daemon=True,
        ).start()

    def _stop_http_flush(self):
        """Cancel the periodic flush timer."""
        if self._http_flush_timer:
            self._http_flush_timer.cancel()
            self._http_flush_timer = None

    # ── Checkpoint upload ────────────────────────────────────────────────

    def upload_checkpoint(self, checkpoint_path: str):
        """Upload a checkpoint file to the dashboard API (no-op if HTTP disabled)."""
        if not self._http_enabled:
            return
        if not os.path.isfile(checkpoint_path):
            logger.warning("Checkpoint file not found for upload: %s", checkpoint_path)
            return
        filename = os.path.basename(checkpoint_path)
        t = threading.Thread(
            target=self._upload_file,
            args=(checkpoint_path, filename),
            daemon=True,
        )
        t.start()

    def _upload_file(self, filepath: str, filename: str, retries: int = 3):
        """Stream a file to the dashboard checkpoint endpoint with retries.

        Files larger than UPLOAD_CHUNK_SIZE are split into multiple requests
        to stay under Cloudflare's 100MB body size limit.
        """
        file_size = os.path.getsize(filepath)
        if file_size <= UPLOAD_CHUNK_SIZE:
            self._upload_single(filepath, filename, file_size, retries)
        else:
            total_chunks = math.ceil(file_size / UPLOAD_CHUNK_SIZE)
            logger.info(
                "Uploading %s (%d bytes) in %d chunks",
                filename, file_size, total_chunks,
            )
            for chunk_idx in range(total_chunks):
                offset = chunk_idx * UPLOAD_CHUNK_SIZE
                length = min(UPLOAD_CHUNK_SIZE, file_size - offset)
                self._upload_chunk(
                    filepath, filename, chunk_idx, total_chunks,
                    offset, length, retries,
                )

    def _upload_single(self, filepath: str, filename: str, file_size: int, retries: int):
        """Upload a complete file in a single request."""
        url = (
            f"{self._http_url}/api/workers/{self._http_job_id}/checkpoints/{filename}"
            f"?expId={self.exp_id}"
        )

        last_err = None
        for attempt in range(retries):
            try:
                with open(filepath, "rb") as f:
                    resp = self._http_session.put(
                        url,
                        data=f,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "Content-Length": str(file_size),
                        },
                        timeout=300,
                    )

                if 200 <= resp.status_code < 300:
                    logger.info(
                        "Uploaded checkpoint %s (%d bytes): %s",
                        filename, file_size, resp.text[:200],
                    )
                    return
                elif resp.status_code >= 500 and attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "HTTP %d uploading %s (attempt %d/%d) — retrying in %ds",
                        resp.status_code, filename, attempt + 1, retries, wait,
                    )
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(
                        "HTTP %d uploading checkpoint %s: %s",
                        resp.status_code, filename, resp.text[:200],
                    )
                    return
            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.debug(
                        "Checkpoint upload failed (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, retries, e, wait,
                    )
                    time.sleep(wait)

        logger.warning(
            "Checkpoint upload failed after %d attempts: %s", retries, last_err,
        )

    def _upload_chunk(
        self, filepath: str, filename: str,
        chunk_idx: int, total_chunks: int,
        offset: int, length: int, retries: int,
    ):
        """Upload one chunk of a file."""
        url = (
            f"{self._http_url}/api/workers/{self._http_job_id}/checkpoints/{filename}"
            f"?expId={self.exp_id}&chunk={chunk_idx}&totalChunks={total_chunks}"
        )

        last_err = None
        for attempt in range(retries):
            try:
                with open(filepath, "rb") as f:
                    f.seek(offset)
                    data = f.read(length)

                resp = self._http_session.put(
                    url,
                    data=data,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(length),
                    },
                    timeout=300,
                )

                if 200 <= resp.status_code < 300:
                    logger.info(
                        "Uploaded chunk %d/%d of %s (%d bytes)",
                        chunk_idx + 1, total_chunks, filename, length,
                    )
                    return
                elif resp.status_code >= 500 and attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "HTTP %d uploading chunk %d/%d of %s (attempt %d/%d) — retrying in %ds",
                        resp.status_code, chunk_idx + 1, total_chunks, filename,
                        attempt + 1, retries, wait,
                    )
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(
                        "HTTP %d uploading chunk %d/%d of %s: %s",
                        resp.status_code, chunk_idx + 1, total_chunks, filename,
                        resp.text[:200],
                    )
                    return
            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.debug(
                        "Chunk upload failed (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, retries, e, wait,
                    )
                    time.sleep(wait)

        logger.warning(
            "Chunk %d/%d upload failed after %d attempts: %s",
            chunk_idx + 1, total_chunks, retries, last_err,
        )

    def _report_experiment_id(self):
        """Tell the dashboard API which experiment this job belongs to."""
        url = f"{self._http_url}/api/workers/{self._http_job_id}/experiment-id"
        try:
            resp = self._http_session.patch(
                url,
                json={"experimentId": self.exp_id},
                timeout=30,
            )
            resp.raise_for_status()
            logger.info("Reported experiment ID %s for job %s", self.exp_id, self._http_job_id)
        except Exception as e:
            logger.warning("Failed to report experiment ID: %s", e)

    # ── File I/O ─────────────────────────────────────────────────────────

    def _initialize_files(self, reset_metrics: bool = True):
        if reset_metrics:
            open(self.metrics_file, "w").close()
        self._update_status({
            "status": "initialized",
            "start_time": time.time(),
            "last_update": time.time(),
        })
        if self._http_enabled:
            self._report_experiment_id()

    def _update_status(self, status_data: dict):
        with open(self.status_file, "w") as f:
            json.dump(status_data, f, indent=2)

        def _write(r):
            r.set(
                f"{self._redis_prefix}:status:{self.exp_id}",
                json.dumps(status_data),
                ex=900,
            )
            r.sadd(f"{self._redis_prefix}:experiments", self.exp_id)
        self._redis_write(_write)

        if self._http_enabled:
            self._http_last_status = status_data

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
                pipe.set(f"{self._redis_prefix}:metrics:{self.exp_id}:latest", serialized, ex=600)
                pipe.rpush(f"{self._redis_prefix}:metrics:{self.exp_id}:history", serialized)
                pipe.ltrim(f"{self._redis_prefix}:metrics:{self.exp_id}:history", -50000, -1)
                pipe.execute()
            self._redis_write(_write)

            self._enqueue_http(data_point)

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
                r.set(f"{self._redis_prefix}:rl:{self.exp_id}:latest", serialized, ex=600)
            self._redis_write(_write)

            self._enqueue_rl_http(data_point)

    def finalize(self):
        """Mark training as complete."""
        with self.lock:
            self._update_status({
                "status": "completed",
                "end_time": time.time(),
                "total_steps": self.metrics_history[-1]["step"] if self.metrics_history else 0,
            })
            self._flush_http_batch()
            self._stop_http_flush()

    def get_metrics_history(self, last_n: int = None) -> list:
        with self.lock:
            if last_n:
                return list(self.metrics_history)[-last_n:]
            return list(self.metrics_history)
