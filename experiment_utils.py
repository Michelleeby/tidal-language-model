"""Shared experiment utilities used by Main.py and train_rl.py."""

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import torch
import redis as redis_lib

VALID_EXPERIMENT_TYPES = ("lm", "rl")


def create_experiment_metadata(
    experiment_type: str,
    source_experiment_id: str | None = None,
    source_checkpoint: str | None = None,
) -> dict:
    """Create a metadata dict for a new experiment.

    Args:
        experiment_type: "lm" or "rl".
        source_experiment_id: For RL experiments, the ID of the source LM experiment.
        source_checkpoint: For RL experiments, the path to the source LM checkpoint.

    Returns:
        A dict suitable for writing to metadata.json.
    """
    if experiment_type not in VALID_EXPERIMENT_TYPES:
        raise ValueError(
            f"Invalid experiment type '{experiment_type}'. Must be one of {VALID_EXPERIMENT_TYPES}"
        )

    return {
        "type": experiment_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_experiment_id": source_experiment_id,
        "source_checkpoint": source_checkpoint,
    }


def write_experiment_metadata(experiment_dir: str, metadata: dict) -> None:
    """Write metadata.json into an experiment directory."""
    path = os.path.join(experiment_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def read_experiment_metadata(experiment_dir: str) -> dict | None:
    """Read metadata.json from an experiment directory.

    Returns None if the file doesn't exist or is invalid JSON.
    """
    path = os.path.join(experiment_dir, "metadata.json")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_git_commit_hash() -> str:
    """Return the short git commit hash, or 'nogit' if unavailable."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .strip()
            .decode("utf-8")
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def get_file_hash(filepath: str) -> str:
    """Return the first 10 hex chars of the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:10]


def resolve_device(config: dict) -> torch.device:
    """Resolve the training device from config, defaulting to auto-detection."""
    device_str = config.get("DEVICE", "auto")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def get_preassigned_experiment_id() -> str | None:
    """Return the pre-assigned experiment ID from the environment, or None.

    When the dashboard pre-creates an experiment before spawning the training
    subprocess, it passes the ID via the TIDAL_EXPERIMENT_ID environment
    variable.  Training scripts should check this before generating their own.
    """
    value = os.environ.get("TIDAL_EXPERIMENT_ID", "")
    return value if value else None


def report_experiment_id_to_job(experiment_id: str) -> None:
    """Report the experiment ID back to the Redis job record.

    Only does something when the TIDAL_JOB_ID environment variable is set
    (i.e. the process was launched by the worker agent).
    """
    job_id = os.environ.get("TIDAL_JOB_ID")
    if not job_id:
        return

    try:
        r = redis_lib.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True,
        )
        job_raw = r.hget("tidal:jobs", job_id)
        if job_raw:
            job = json.loads(job_raw)
            job["experimentId"] = experiment_id
            job["updatedAt"] = time.time()
            r.hset("tidal:jobs", job_id, json.dumps(job))
            r.publish("tidal:job:updates", json.dumps({"jobId": job_id}))
    except Exception:
        pass
