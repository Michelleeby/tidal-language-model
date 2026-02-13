import http.client
import json
import os
import time
import logging
import threading
import urllib.request
import urllib.error
from collections import deque
from urllib.parse import urlparse


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

    def __init__(self, experiment_dir: str, max_history: int = 10000, reset_metrics: bool = True):
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

        # HTTP forwarding for remote workers (vast.ai → dashboard API)
        self._http_url = os.environ.get("TIDAL_API_URL")
        self._http_token = os.environ.get("TIDAL_AUTH_TOKEN")
        self._http_job_id = os.environ.get("TIDAL_JOB_ID")
        self._http_enabled = bool(
            self._http_url and self._http_token and self._http_job_id
        )
        self._http_batch: list[dict] = []
        self._http_last_status: dict | None = None
        self._http_flush_timer: threading.Timer | None = None
        if self._http_enabled:
            self._http_url = self._http_url.rstrip("/")
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

    def _http_forward(self, body: dict, retries: int = 3):
        """POST metrics to the dashboard API with exponential backoff retry."""
        url = f"{self._http_url}/api/workers/{self._http_job_id}/metrics"
        data = json.dumps(body).encode()
        last_err = None

        for attempt in range(retries):
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Authorization", f"Bearer {self._http_token}")
            req.add_header("Content-Type", "application/json")
            req.add_header("User-Agent", "TidalMetrics/1.0")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp.read()
                    return
            except urllib.error.HTTPError as e:
                logger.warning("HTTP %d forwarding metrics: %s", e.code, e.read().decode()[:200])
                return  # don't retry HTTP errors
            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.debug("Metrics POST failed (attempt %d/%d): %s — retrying in %ds", attempt + 1, retries, e, wait)
                    time.sleep(wait)

        logger.warning("Metrics POST failed after %d attempts: %s", retries, last_err)

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
            import math
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
        parsed = urlparse(self._http_url)
        use_https = parsed.scheme == "https"
        host = parsed.hostname
        port = parsed.port or (443 if use_https else 80)

        url_path = (
            f"/api/workers/{self._http_job_id}/checkpoints/{filename}"
            f"?expId={self.exp_id}"
        )

        last_err = None
        for attempt in range(retries):
            try:
                if use_https:
                    conn = http.client.HTTPSConnection(host, port, timeout=300)
                else:
                    conn = http.client.HTTPConnection(host, port, timeout=300)

                conn.putrequest("PUT", url_path)
                conn.putheader("Authorization", f"Bearer {self._http_token}")
                conn.putheader("Content-Type", "application/octet-stream")
                conn.putheader("Content-Length", str(file_size))
                conn.putheader("User-Agent", "TidalMetrics/1.0")
                conn.endheaders()

                with open(filepath, "rb") as f:
                    while True:
                        data = f.read(1024 * 1024)
                        if not data:
                            break
                        conn.send(data)

                resp = conn.getresponse()
                body = resp.read()
                conn.close()

                if 200 <= resp.status < 300:
                    logger.info(
                        "Uploaded checkpoint %s (%d bytes): %s",
                        filename, file_size, body.decode()[:200],
                    )
                    return
                elif resp.status >= 500 and attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "HTTP %d uploading %s (attempt %d/%d) — retrying in %ds",
                        resp.status, filename, attempt + 1, retries, wait,
                    )
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(
                        "HTTP %d uploading checkpoint %s: %s",
                        resp.status, filename, body.decode()[:200],
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
        parsed = urlparse(self._http_url)
        use_https = parsed.scheme == "https"
        host = parsed.hostname
        port = parsed.port or (443 if use_https else 80)

        url_path = (
            f"/api/workers/{self._http_job_id}/checkpoints/{filename}"
            f"?expId={self.exp_id}&chunk={chunk_idx}&totalChunks={total_chunks}"
        )

        last_err = None
        for attempt in range(retries):
            try:
                if use_https:
                    conn = http.client.HTTPSConnection(host, port, timeout=300)
                else:
                    conn = http.client.HTTPConnection(host, port, timeout=300)

                conn.putrequest("PUT", url_path)
                conn.putheader("Authorization", f"Bearer {self._http_token}")
                conn.putheader("Content-Type", "application/octet-stream")
                conn.putheader("Content-Length", str(length))
                conn.putheader("User-Agent", "TidalMetrics/1.0")
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
                body = resp.read()
                conn.close()

                if 200 <= resp.status < 300:
                    logger.info(
                        "Uploaded chunk %d/%d of %s (%d bytes)",
                        chunk_idx + 1, total_chunks, filename, length,
                    )
                    return
                elif resp.status >= 500 and attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "HTTP %d uploading chunk %d/%d of %s (attempt %d/%d) — retrying in %ds",
                        resp.status, chunk_idx + 1, total_chunks, filename,
                        attempt + 1, retries, wait,
                    )
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(
                        "HTTP %d uploading chunk %d/%d of %s: %s",
                        resp.status, chunk_idx + 1, total_chunks, filename,
                        body.decode()[:200],
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
        data = json.dumps({"experimentId": self.exp_id}).encode()
        req = urllib.request.Request(url, data=data, method="PATCH")
        req.add_header("Authorization", f"Bearer {self._http_token}")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "TidalMetrics/1.0")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
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
                f"tidal:status:{self.exp_id}",
                json.dumps(status_data),
                ex=900,
            )
            r.sadd("tidal:experiments", self.exp_id)
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
                pipe.set(f"tidal:metrics:{self.exp_id}:latest", serialized, ex=600)
                pipe.rpush(f"tidal:metrics:{self.exp_id}:history", serialized)
                pipe.ltrim(f"tidal:metrics:{self.exp_id}:history", -50000, -1)
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
                r.set(f"tidal:rl:{self.exp_id}:latest", serialized, ex=600)
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
