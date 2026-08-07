"""Read-only environment probes and scheduler adapters."""

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


def _run(argv: List[str], timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout, check=False)


def cuda_visible_ids() -> Optional[List[str]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    if not raw.strip() or raw.strip() in ("-1", "NoDevFiles"):
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def query_nvidia() -> List[Dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    result = _run([executable, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
    if result.returncode != 0:
        return []
    devices = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3:
            try:
                memory = int(parts[2])
            except ValueError:
                memory = None
            devices.append({"id": parts[0], "name": parts[1], "memory_mib": memory})
    return devices


def slurm_available() -> bool:
    return all(shutil.which(command) for command in ("sbatch", "squeue", "sacct", "scancel"))


def resolve_mode(config: Dict[str, Any]) -> str:
    requested = config.get("mode", "auto")
    if requested != "auto":
        if requested == "slurm" and not slurm_available():
            raise RuntimeError("slurm mode requested but required commands are unavailable")
        return str(requested)
    # An interactive/batch allocation is already scheduler-isolated; do not submit nested jobs.
    if os.environ.get("SLURM_JOB_ID"):
        return "direct"
    return "slurm" if slurm_available() else "direct"


def effective_allowed_gpu_ids(config: Dict[str, Any], mode: str) -> List[str]:
    visible = cuda_visible_ids()
    configured = [str(item) for item in config.get("allowed_gpu_ids", [])]
    if visible is not None:
        # CUDA_VISIBLE_DEVICES is an absolute boundary. Config may only narrow it.
        if configured:
            return [item for item in configured if item in visible][:4]
        return visible[:4]
    if configured:
        return configured[:4]
    if mode == "direct":
        return [str(device["id"]) for device in query_nvidia()[:4]]
    # On a Slurm login node, concrete device IDs are assigned only inside the job.
    return []


def torch_info() -> Dict[str, Any]:
    try:
        import torch  # type: ignore
        return {
            "installed": True,
            "version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:
        return {"installed": False, "error": "%s: %s" % (type(exc).__name__, exc)}


def scheduler_inventory() -> List[str]:
    if not shutil.which("sinfo"):
        return []
    result = _run(["sinfo", "-h", "-o", "%P|%a|%G|%D|%t"])
    if result.returncode != 0:
        return []
    return sorted(set(line.strip() for line in result.stdout.splitlines() if line.strip()))


def environment_snapshot(project_root: Path, mode: str) -> Dict[str, Any]:
    return {
        "hostname": platform.node(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "pytorch": torch_info(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": query_nvidia(),
        "scheduler": mode,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "project_path": str(project_root),
    }


def slurm_state(job_id: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Return harness state, exit code, and raw scheduler state."""
    squeue = shutil.which("squeue")
    if squeue:
        result = _run([squeue, "-h", "-j", str(job_id), "-o", "%T"])
        raw = result.stdout.strip().splitlines()
        if result.returncode == 0 and raw:
            state = raw[0].strip().upper()
            if state in ("PENDING", "CONFIGURING", "RESV_DEL_HOLD", "REQUEUE_FED", "REQUEUED"):
                return "QUEUED", None, state
            if state in ("RUNNING", "COMPLETING", "STAGE_OUT", "SIGNALING"):
                return "RUNNING", None, state
            if state in ("CANCELLED", "PREEMPTED"):
                return "STOPPED", None, state
            if state == "COMPLETED":
                return "SUCCEEDED", 0, state
            if state in ("FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"):
                return "FAILED", None, state
    sacct = shutil.which("sacct")
    if not sacct:
        return None, None, None
    result = _run([sacct, "-n", "-X", "-j", str(job_id), "--format=State,ExitCode", "--parsable2"])
    if result.returncode != 0:
        return None, None, None
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2 or not parts[0]:
            continue
        raw_state = parts[0].split()[0].split("+")[0].upper()
        try:
            code = int(parts[1].split(":", 1)[0])
        except (ValueError, IndexError):
            code = None
        if raw_state == "COMPLETED":
            return "SUCCEEDED", 0 if code is None else code, raw_state
        if raw_state in ("CANCELLED", "PREEMPTED"):
            return "STOPPED", code, raw_state
        if raw_state in ("PENDING", "CONFIGURING", "REQUEUED"):
            return "QUEUED", None, raw_state
        if raw_state in ("RUNNING", "COMPLETING"):
            return "RUNNING", None, raw_state
        return "FAILED", code, raw_state
    return None, None, None
