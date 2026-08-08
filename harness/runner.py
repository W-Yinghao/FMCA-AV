"""Internal job wrapper. Users should invoke harness.cli, not this module."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any, Dict, Optional

from . import scheduler
from .state import PROJECT_ROOT, atomic_write_json, load_config, locked, now_iso, read_json, runs_dir, update_job


_child: Optional[subprocess.Popen] = None
_stopping = False


def _handle_stop(signum: int, _frame: Any) -> None:
    global _stopping
    _stopping = True
    child = _child
    if child is not None and child.poll() is None:
        try:
            child.send_signal(signum)
        except ProcessLookupError:
            pass


def _write_environment(path: Path, info: Dict[str, Any]) -> None:
    lines = []
    for key in ("hostname", "python_executable", "python_version", "scheduler", "slurm_job_id",
                "cuda_visible_devices", "project_path"):
        lines.append("%s: %s" % (key, info.get(key)))
    lines.append("pytorch: %s" % json.dumps(info.get("pytorch"), ensure_ascii=False, sort_keys=True))
    lines.append("harness_python: %s" % json.dumps(info.get("harness_python"), ensure_ascii=False, sort_keys=True))
    lines.append("workload_python: %s" % json.dumps(info.get("workload_python"), ensure_ascii=False, sort_keys=True))
    lines.append("runtime_environment: %s" % json.dumps(info.get("runtime_environment"), ensure_ascii=False, sort_keys=True))
    lines.append("gpus: %s" % json.dumps(info.get("gpus"), ensure_ascii=False, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(run_id: str) -> int:
    global _child
    config = load_config()
    run_dir = runs_dir(config) / run_id
    request = read_json(run_dir / "request.json")
    mode = str(request["resolved_mode"])
    visible = scheduler.cuda_visible_ids()
    requested_gpus = int(request["requested_gpus"])
    if mode == "direct":
        actual = list(request.get("allocated_gpu_ids", []))
    else:
        actual = visible[:requested_gpus] if requested_gpus and visible is not None else []
    env = os.environ.copy()
    env["FMCA_HARNESS_RUN_ID"] = run_id
    env["FMCA_HARNESS_RUN_DIR"] = str(run_dir)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    for key, value in request.get("environment", {}).items():
        env[str(key)] = str(value)
    if mode == "slurm":
        # Never replace the scheduler-provided boundary with login-node placeholders.
        if requested_gpus == 0:
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif visible is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(visible[:requested_gpus])
    command = [str(item) for item in request["final_command"]]
    with locked():
        update_job(run_id, {
            "state": "RUNNING",
            "actual_gpu_ids": actual,
            "pid": os.getpid(),
            "start_time": now_iso(),
            "failure_reason": None,
        }, config)

    snapshot_command = command
    if bool(request.get("local_watcher", False)):
        # Watchers only orchestrate Slurm children. Avoid importing the full
        # PyTorch stack on the login node just to snapshot an idle controller.
        snapshot_command = []
    elif scheduler.infer_workload_python(command) is None and config.get("workload_python"):
        snapshot_command = [str(config["workload_python"])]
    env_info = scheduler.environment_snapshot(PROJECT_ROOT, mode, snapshot_command, env)
    env_info["cuda_visible_devices"] = env.get("CUDA_VISIBLE_DEVICES")
    env_info["runtime_environment"] = {
        key: env.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONUNBUFFERED", "PYTHONFAULTHANDLER",
        )
    }
    _write_environment(run_dir / "environment.txt", env_info)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    try:
        _child = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=env)
        exit_code = _child.wait()
        stopped = _stopping or exit_code in (-signal.SIGTERM, -signal.SIGKILL, -signal.SIGINT, 128 + signal.SIGTERM)
        final_state = "STOPPED" if stopped else ("SUCCEEDED" if exit_code == 0 else "FAILED")
        reason = None if final_state == "SUCCEEDED" else (
            "received stop signal" if final_state == "STOPPED" else "command exited with code %s" % exit_code)
    except Exception as exc:
        exit_code = 127
        final_state = "FAILED"
        reason = "runner could not start command: %s: %s" % (type(exc).__name__, exc)
    with locked():
        update_job(run_id, {
            "state": final_state,
            "end_time": now_iso(),
            "exit_code": exit_code,
            "failure_reason": reason,
        }, config)
    return int(exit_code or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description="internal harness job runner")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    return run(args.run)


if __name__ == "__main__":
    sys.exit(main())
