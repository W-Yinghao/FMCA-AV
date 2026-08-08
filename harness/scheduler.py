"""Read-only environment probes and scheduler adapters."""

import json
import os
from pathlib import Path
import platform
import re
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
    return all(shutil.which(command) for command in ("sbatch", "squeue", "scancel"))


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
            return [item for item in configured if item in visible][:6]
        return visible[:6]
    if configured:
        return configured[:6]
    if mode == "direct":
        return [str(device["id"]) for device in query_nvidia()[:6]]
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


def infer_workload_python(command: Optional[List[str]]) -> Optional[str]:
    """Find an explicit Python executable in a command without executing it."""
    if not command:
        return None
    python_name = re.compile(r"^(python|python3)(?:\.[0-9]+)*$")
    for token in command:
        value = str(token)
        if python_name.fullmatch(Path(value).name):
            if Path(value).is_absolute():
                return value
            resolved = shutil.which(value)
            if resolved:
                return resolved
    # Project torchrun is a small shell wrapper around the environment Python.
    # Reading its command line is safer than importing the workload environment
    # into the harness process.
    first = Path(str(command[0]))
    candidate = first if first.is_absolute() else Path.cwd() / first
    if candidate.is_file() and candidate.stat().st_size <= 65536:
        try:
            contents = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            contents = ""
        match = re.search(r"(?m)(/[A-Za-z0-9_./-]+/python(?:[0-9.]*)?)\s", contents)
        if match:
            return match.group(1)
    return None


def python_runtime_info(executable: str, environment: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Probe a workload interpreter in a subprocess so its torch is reported."""
    probe = (
        "import importlib.metadata as md\n"
        "import importlib.util,json,platform,sys\n"
        "r={'executable':sys.executable,'version':platform.python_version()}\n"
        "r['packages']={}\n"
        "for p in ('torch','torchvision','lightning'):\n"
        " try:\n"
        "  s=importlib.util.find_spec(p); r['packages'][p]={'version':md.version(p),'origin':s.origin if s else None}\n"
        " except Exception as e:\n"
        "  r['packages'][p]={'error':type(e).__name__+': '+str(e)}\n"
        "try:\n"
        " import torch\n"
        " r['pytorch']={'installed':True,'version':str(torch.__version__),"
        "'cuda_version':str(torch.version.cuda),'cuda_available':bool(torch.cuda.is_available()),"
        "'device_count':int(torch.cuda.device_count()),'origin':str(torch.__file__)}\n"
        "except Exception as e:\n"
        " r['pytorch']={'installed':False,'error':type(e).__name__+': '+str(e)}\n"
        "print(json.dumps(r,sort_keys=True))\n"
    )
    try:
        result = subprocess.run(
            [executable, "-c", probe], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30, check=False, env=environment,
        )
        if result.returncode != 0:
            return {
                "requested_executable": executable,
                "probe_error": "exit %d: %s" % (result.returncode, result.stderr.strip()[-1000:]),
            }
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        if not isinstance(payload, dict):
            raise ValueError("probe did not return an object")
        return payload
    except Exception as exc:
        return {
            "requested_executable": executable,
            "probe_error": "%s: %s" % (type(exc).__name__, exc),
        }


def scheduler_inventory() -> List[str]:
    if not shutil.which("sinfo"):
        return []
    result = _run(["sinfo", "-h", "-o", "%P|%a|%G|%D|%t"])
    if result.returncode != 0:
        return []
    return sorted(set(line.strip() for line in result.stdout.splitlines() if line.strip()))


def active_user_slurm_jobs() -> Optional[int]:
    """Count active jobs visible to squeue, expanding array elements."""
    squeue = shutil.which("squeue")
    user = os.environ.get("USER")
    if not squeue or not user:
        return None
    result = _run([squeue, "-h", "-r", "-u", user, "-o", "%i"])
    if result.returncode != 0:
        return None
    return sum(1 for line in result.stdout.splitlines() if line.strip())


def environment_snapshot(
    project_root: Path,
    mode: str,
    workload_command: Optional[List[str]] = None,
    environment: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    workload_executable = infer_workload_python(workload_command)
    workload = python_runtime_info(workload_executable, environment) if workload_executable else None
    primary_executable = str(workload.get("executable")) if workload and workload.get("executable") else sys.executable
    primary_version = str(workload.get("version")) if workload and workload.get("version") else platform.python_version()
    primary_torch = workload.get("pytorch") if workload and workload.get("pytorch") else torch_info()
    return {
        "hostname": platform.node(),
        "python_executable": primary_executable,
        "python_version": primary_version,
        "pytorch": primary_torch,
        "harness_python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "pytorch": torch_info(),
        },
        "workload_python": workload,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": query_nvidia(),
        "scheduler": mode,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "project_path": str(project_root),
    }


def slurm_state(job_id: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Return harness state using squeue as the only scheduler status source."""
    squeue = shutil.which("squeue")
    if not squeue:
        return None, None, None
    result = _run([squeue, "-h", "-j", str(job_id), "-o", "%T"])
    if result.returncode != 0:
        return None, None, None
    raw = result.stdout.strip().splitlines()
    if not raw:
        # A normal harness runner writes its terminal state before leaving Slurm.
        # If the scheduler no longer knows the job but its file state is active,
        # the runner disappeared and refresh should repair that stale state.
        return "FAILED", None, "NOT_IN_SQUEUE"
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
    return None, None, state


def slurm_states(job_ids: List[str]) -> Dict[str, Tuple[Optional[str], Optional[int], Optional[str]]]:
    """Batch form of slurm_state using one squeue invocation."""
    unique = list(dict.fromkeys(str(job_id) for job_id in job_ids if str(job_id)))
    if not unique:
        return {}
    squeue = shutil.which("squeue")
    if not squeue:
        return {job_id: (None, None, None) for job_id in unique}
    user = os.environ.get("USER")
    argv = [squeue, "-h"]
    if user:
        argv.extend(["-u", user])
    argv.extend(["-o", "%i|%T"])
    result = _run(argv)
    if result.returncode != 0:
        return {job_id: (None, None, None) for job_id in unique}
    raw_states: Dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|", 1)
        if len(fields) == 2:
            raw_states[fields[0].strip()] = fields[1].strip().upper()
    mapped: Dict[str, Tuple[Optional[str], Optional[int], Optional[str]]] = {}
    for job_id in unique:
        raw = raw_states.get(job_id)
        if raw is None:
            mapped[job_id] = ("FAILED", None, "NOT_IN_SQUEUE")
        elif raw in ("PENDING", "CONFIGURING", "RESV_DEL_HOLD", "REQUEUE_FED", "REQUEUED"):
            mapped[job_id] = ("QUEUED", None, raw)
        elif raw in ("RUNNING", "COMPLETING", "STAGE_OUT", "SIGNALING"):
            mapped[job_id] = ("RUNNING", None, raw)
        elif raw in ("CANCELLED", "PREEMPTED"):
            mapped[job_id] = ("STOPPED", None, raw)
        elif raw == "COMPLETED":
            mapped[job_id] = ("SUCCEEDED", 0, raw)
        elif raw in ("FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"):
            mapped[job_id] = ("FAILED", None, raw)
        else:
            mapped[job_id] = (None, None, raw)
    return mapped
