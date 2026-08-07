"""Configuration, locking, and atomic JSON state primitives."""

import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Iterator, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = PROJECT_ROOT / "harness"
CONFIG_PATH = HARNESS_DIR / "config.json"
STATE_DIR = HARNESS_DIR / "state"
JOBS_PATH = STATE_DIR / "jobs.json"
LOCK_PATH = STATE_DIR / "harness.lock"
ALLOWED_GPU_COUNTS = (0, 1, 2, 4)
ACTIVE_STATES = ("QUEUED", "RUNNING")
FINAL_STATES = ("SUCCEEDED", "FAILED", "STOPPED", "BLOCKED")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path: Path, default: Optional[Any] = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        if default is not None:
            return default
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@contextlib.contextmanager
def locked() -> Iterator[None]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_config() -> Dict[str, Any]:
    config = read_json(CONFIG_PATH)
    if config.get("max_gpus") != 4:
        raise ValueError("config max_gpus must be exactly 4")
    if config.get("mode") not in ("auto", "direct", "slurm"):
        raise ValueError("config mode must be auto, direct, or slurm")
    allowed = config.get("allowed_gpu_ids")
    if not isinstance(allowed, list) or len(allowed) > 4:
        raise ValueError("allowed_gpu_ids must be a list of at most four IDs")
    if len({str(item) for item in allowed}) != len(allowed):
        raise ValueError("allowed_gpu_ids contains duplicates")
    if int(config.get("stop_grace_seconds", 30)) < 0:
        raise ValueError("stop_grace_seconds must be non-negative")
    if int(config.get("omp_num_threads", 1)) < 1:
        raise ValueError("omp_num_threads must be positive")
    slurm_args = config.get("slurm_args", [])
    if not isinstance(slurm_args, list) or not all(isinstance(item, str) for item in slurm_args):
        raise ValueError("slurm_args must be a list of strings")
    forbidden = ("--gres", "--gpus", "--gpus-per-", "--nodes", "-N", "--ntasks", "-n",
                 "--output", "-o", "--error", "-e", "--wrap", "--job-name", "-J")
    for item in slurm_args:
        if any(item == prefix or item.startswith(prefix + "=") or item.startswith("--gpus-per-")
               for prefix in forbidden):
            raise ValueError("slurm_args may not override managed option: %s" % item)
    partitions = config.get("slurm_partitions", {})
    required_partition_sets = ("cpu", "default_gpu", "imagenet_gpu")
    if not isinstance(partitions, dict):
        raise ValueError("slurm_partitions must be an object")
    for key in required_partition_sets:
        values = partitions.get(key)
        if (not isinstance(values, list) or not values or
                not all(isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", item)
                        for item in values)):
            raise ValueError("slurm_partitions.%s must be a non-empty list of partition names" % key)
    return config


def runs_dir(config: Dict[str, Any]) -> Path:
    configured = Path(str(config.get("runs_dir", "runs")))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def load_jobs() -> Dict[str, Any]:
    data = read_json(JOBS_PATH, {"jobs": {}})
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict):
        raise ValueError("harness/state/jobs.json has an invalid structure")
    return data


def save_jobs(data: Dict[str, Any]) -> None:
    atomic_write_json(JOBS_PATH, data)


def status_path(run_id: str, config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = config or load_config()
    return runs_dir(cfg) / run_id / "status.json"


def update_job(run_id: str, updates: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Update global and per-run state. Caller must hold locked()."""
    cfg = config or load_config()
    jobs = load_jobs()
    if run_id not in jobs["jobs"]:
        raise KeyError("unknown run: %s" % run_id)
    jobs["jobs"][run_id].update(updates)
    record = jobs["jobs"][run_id]
    save_jobs(jobs)
    atomic_write_json(status_path(run_id, cfg), record)
    return record


def pid_alive(pid: Any) -> bool:
    try:
        number = int(pid)
        os.kill(number, 0)
        stat_path = Path("/proc") / str(number) / "stat"
        if stat_path.exists():
            fields = stat_path.read_text(encoding="utf-8").split()
            if len(fields) > 2 and fields[2] == "Z":
                return False
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError, OSError):
        return False


def pid_is_harness_run(pid: Any, run_id: str) -> bool:
    if not pid_alive(pid):
        return False
    try:
        command = (Path("/proc") / str(int(pid)) / "cmdline").read_bytes().split(b"\0")
        decoded = [item.decode("utf-8", errors="replace") for item in command if item]
        return "harness.runner" in decoded and run_id in decoded
    except (OSError, TypeError, ValueError):
        return False
