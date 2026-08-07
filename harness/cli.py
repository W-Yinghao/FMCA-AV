"""Command-line interface for the FMCA-AV experiment harness."""

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import scheduler
from .state import (ACTIVE_STATES, ALLOWED_GPU_COUNTS, CONFIG_PATH, FINAL_STATES,
                    PROJECT_ROOT, atomic_write_json, load_config, load_jobs, locked,
                    now_iso, pid_alive, read_json, runs_dir, save_jobs, update_job)
from .state import pid_is_harness_run


def fail(message: str, code: int = 2) -> int:
    print("error: " + message, file=sys.stderr)
    return code


def validate_gpus(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("GPU count must be one of 0, 1, 2, 4")
    if count not in ALLOWED_GPU_COUNTS:
        raise argparse.ArgumentTypeError("GPU count must be one of 0, 1, 2, 4")
    return count


def normalize_command(raw: Sequence[str]) -> List[str]:
    command = list(raw)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")
    return command


def _torchrun_nproc(command: Sequence[str]) -> Optional[int]:
    for index, value in enumerate(command):
        if value.startswith("--nproc_per_node="):
            try:
                return int(value.split("=", 1)[1])
            except ValueError:
                return None
        if value == "--nproc_per_node" and index + 1 < len(command):
            try:
                return int(command[index + 1])
            except ValueError:
                return None
    return None


def _torchrun_nnodes(command: Sequence[str]) -> int:
    for index, value in enumerate(command):
        if value.startswith("--nnodes="):
            try:
                return int(value.split("=", 1)[1])
            except ValueError:
                return -1
        if value == "--nnodes" and index + 1 < len(command):
            try:
                return int(command[index + 1])
            except ValueError:
                return -1
    return 1


def build_command(command: Sequence[str], gpus: int) -> List[str]:
    command = [str(item) for item in command]
    if gpus in (0, 1):
        return command
    first = Path(command[0]).name
    if first == "torchrun":
        nproc = _torchrun_nproc(command)
        if nproc is None:
            raise ValueError("explicit torchrun commands must specify --nproc_per_node")
        if nproc != gpus:
            raise ValueError("torchrun --nproc_per_node=%s does not match --gpus %s" % (nproc, gpus))
        if _torchrun_nnodes(command) != 1:
            raise ValueError("multi-node torchrun is not supported")
        return command
    python_names = {"python", "python3", Path(sys.executable).name}
    if first in python_names and len(command) >= 2 and command[1].endswith(".py"):
        return ["torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=%d" % gpus] + command[1:]
    raise ValueError("2/4-GPU commands must be explicit torchrun, or 'python <script>.py ...'")


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-._")
    if not cleaned:
        raise ValueError("name must contain at least one letter or digit")
    return cleaned[:80]


def new_run_id(name: str, base: Path) -> str:
    prefix = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S") + "_" + _safe_name(name)
    candidate = prefix
    suffix = 2
    while (base / candidate).exists():
        candidate = "%s_%d" % (prefix, suffix)
        suffix += 1
    return candidate


def _write_environment(path: Path, info: Dict[str, Any]) -> None:
    lines = []
    for key in ("hostname", "python_executable", "python_version", "scheduler", "slurm_job_id",
                "cuda_visible_devices", "project_path"):
        lines.append("%s: %s" % (key, info.get(key)))
    lines.append("pytorch: %s" % json.dumps(info.get("pytorch"), ensure_ascii=False, sort_keys=True))
    lines.append("gpus: %s" % json.dumps(info.get("gpus"), ensure_ascii=False, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_project_candidates() -> Tuple[List[str], List[str]]:
    code = []
    data = []
    ignored = {"harness", "runs", ".git", "__pycache__"}
    entry_names = {"train.py", "main.py", "run.py", "pretrain.py", "evaluate.py", "eval.py"}
    data_names = {"data", "dataset", "datasets", "imagenet", "imagenet100", "cifar", "cifar10", "cifar100"}
    for root, dirs, files in os.walk(str(PROJECT_ROOT)):
        relative = Path(root).relative_to(PROJECT_ROOT)
        if len(relative.parts) >= 4:
            dirs[:] = []
            continue
        dirs[:] = [item for item in dirs if item not in ignored]
        for filename in files:
            if filename.lower() in entry_names or (filename.endswith(".py") and any(
                    token in filename.lower() for token in ("fmca", "hfmca", "train"))):
                code.append(str((Path(root) / filename).relative_to(PROJECT_ROOT)))
        for dirname in dirs:
            if dirname.lower() in data_names:
                data.append(str((Path(root) / dirname).relative_to(PROJECT_ROOT)))
    return sorted(set(code)), sorted(set(data))


def doctor(_args: argparse.Namespace) -> int:
    try:
        config = load_config()
        mode = scheduler.resolve_mode(config)
    except Exception as exc:
        return fail(str(exc))
    allowed = scheduler.effective_allowed_gpu_ids(config, mode)
    info = scheduler.environment_snapshot(PROJECT_ROOT, mode)
    code, data = _find_project_candidates()
    report = {
        "project_path": str(PROJECT_ROOT),
        "config_path": str(CONFIG_PATH),
        "resolved_mode": mode,
        "inside_slurm_allocation": bool(os.environ.get("SLURM_JOB_ID")),
        "max_gpus": config["max_gpus"],
        "allowed_gpu_ids": allowed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "pytorch": info["pytorch"],
        "gpus": info["gpus"],
        "scheduler_inventory": scheduler.scheduler_inventory(),
        "slurm_partitions": config.get("slurm_partitions", {}),
        "code_entry_candidates": code,
        "data_directory_candidates": data,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def allocation_for(config: Dict[str, Any], mode: str, gpus: int,
                   jobs: Dict[str, Any]) -> Tuple[List[str], int]:
    used = sum(int(record.get("requested_gpus", 0)) for record in jobs["jobs"].values()
               if record.get("state") in ACTIVE_STATES)
    if used + gpus > int(config["max_gpus"]):
        raise RuntimeError("GPU limit exceeded: %d active/queued + %d requested > 4" % (used, gpus))
    if gpus == 0 or mode == "slurm":
        return [], used
    allowed = scheduler.effective_allowed_gpu_ids(config, mode)
    occupied = set()
    for record in jobs["jobs"].values():
        if record.get("state") in ACTIVE_STATES:
            occupied.update(str(item) for item in record.get("actual_gpu_ids", []))
    free = [gpu for gpu in allowed if gpu not in occupied]
    if len(free) < gpus:
        raise RuntimeError("only %d allowed direct GPU(s) are free; %d requested" % (len(free), gpus))
    return free[:gpus], used


def _refresh_one_locked(run_id: str, record: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if record.get("state") not in ACTIVE_STATES:
        return record
    mode = record.get("resolved_mode")
    updates: Dict[str, Any] = {}
    if mode == "direct":
        pid = record.get("pid")
        if pid and not pid_is_harness_run(pid, run_id):
            updates = {
                "state": "FAILED",
                "end_time": now_iso(),
                "exit_code": record.get("exit_code"),
                "failure_reason": "stale active state repaired: process %s is not alive" % pid,
            }
    elif mode == "slurm":
        job_id = record.get("slurm_job_id")
        if job_id:
            state, exit_code, raw = scheduler.slurm_state(str(job_id))
            if state and state != record.get("state"):
                updates["state"] = state
                updates["scheduler_state"] = raw
                if state == "RUNNING" and not record.get("start_time"):
                    updates["start_time"] = now_iso()
                if state in FINAL_STATES:
                    updates["end_time"] = record.get("end_time") or now_iso()
                    updates["exit_code"] = exit_code
                    if state != "SUCCEEDED" and not record.get("failure_reason"):
                        updates["failure_reason"] = "Slurm state %s" % raw
    if updates:
        return update_job(run_id, updates, config)
    return record


def refresh_all_locked(config: Dict[str, Any]) -> Dict[str, Any]:
    jobs = load_jobs()
    for run_id in list(jobs["jobs"]):
        current = load_jobs()["jobs"][run_id]
        _refresh_one_locked(run_id, current, config)
    return load_jobs()


def dry_run(args: argparse.Namespace) -> int:
    try:
        config = load_config()
        mode = scheduler.resolve_mode(config)
        raw = normalize_command(args.command)
        final = build_command(raw, args.gpus)
        allocation_error = None
        with locked():
            jobs = refresh_all_locked(config)
            try:
                allocated, used = allocation_for(config, mode, args.gpus, jobs)
            except RuntimeError as exc:
                allocated = []
                used = sum(int(record.get("requested_gpus", 0))
                           for record in jobs["jobs"].values()
                           if record.get("state") in ACTIVE_STATES)
                allocation_error = str(exc)
    except Exception as exc:
        return fail(str(exc))
    environment = {"OMP_NUM_THREADS": str(config.get("omp_num_threads", 1))}
    if mode == "direct" or args.gpus == 0:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(allocated)
    result = {
        "dry_run": True,
        "name": args.name,
        "requested_gpus": args.gpus,
        "active_or_queued_gpus": used,
        "resolved_mode": mode,
        "profile": args.profile,
        "slurm_partitions": selected_partitions(config, args.gpus, args.profile) if mode == "slurm" else [],
        "allocated_gpu_ids": allocated,
        "allocation": "Slurm will assign %d GPU(s)" % args.gpus if mode == "slurm" and args.gpus else allocated,
        "command": final,
        "command_text": shlex.join(final),
        "environment": environment,
        "allocatable": allocation_error is None,
        "allocation_error": allocation_error,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if allocation_error is None else 3


def _create_run_locked(name: str, gpus: int, raw: List[str], final: List[str], mode: str,
                       allocated: List[str], config: Dict[str, Any], retry_from: Optional[str],
                       profile: str) -> Tuple[str, Path]:
    base = runs_dir(config)
    base.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id(name, base)
    run_dir = base / run_id
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    for filename in ("stdout.log", "stderr.log", "metrics.jsonl"):
        (run_dir / filename).touch(exist_ok=False)
    (run_dir / "command.txt").write_text(shlex.join(final) + "\n", encoding="utf-8")
    environment = {"OMP_NUM_THREADS": str(config.get("omp_num_threads", 1))}
    if mode == "direct" or gpus == 0:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(allocated)
    request = {
        "run_id": run_id,
        "name": name,
        "requested_gpus": gpus,
        "original_command": raw,
        "final_command": final,
        "resolved_mode": mode,
        "profile": profile,
        "allocated_gpu_ids": allocated,
        "environment": environment,
        "retry_from": retry_from,
        "created_at": now_iso(),
        "config": config,
    }
    atomic_write_json(run_dir / "request.json", request)
    record = {
        "run_id": run_id,
        "name": name,
        "state": "QUEUED",
        "requested_gpus": gpus,
        "actual_gpu_ids": allocated,
        "pid": None,
        "slurm_job_id": None,
        "resolved_mode": mode,
        "profile": profile,
        "created_at": request["created_at"],
        "start_time": None,
        "end_time": None,
        "exit_code": None,
        "failure_reason": None,
        "retry_from": retry_from,
    }
    jobs = load_jobs()
    jobs["jobs"][run_id] = record
    save_jobs(jobs)
    atomic_write_json(run_dir / "status.json", record)
    _write_environment(run_dir / "environment.txt", scheduler.environment_snapshot(PROJECT_ROOT, mode))
    return run_id, run_dir


def _launch_direct(run_id: str, run_dir: Path, config: Dict[str, Any]) -> None:
    stdout_handle = (run_dir / "stdout.log").open("ab", buffering=0)
    stderr_handle = (run_dir / "stderr.log").open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "harness.runner", "--run", run_id],
            cwd=str(PROJECT_ROOT), stdout=stdout_handle, stderr=stderr_handle,
            start_new_session=True, close_fds=True)
    finally:
        stdout_handle.close()
        stderr_handle.close()
    update_job(run_id, {"pid": process.pid}, config)


def selected_partitions(config: Dict[str, Any], gpus: int, profile: str) -> List[str]:
    key = "cpu" if gpus == 0 else ("imagenet_gpu" if profile == "imagenet" else "default_gpu")
    return [str(item) for item in config["slurm_partitions"][key]]


def _launch_slurm(run_id: str, run_dir: Path, gpus: int, config: Dict[str, Any], profile: str) -> None:
    script = run_dir / "launch.slurm"
    body = "#!/bin/bash\nset -eu\ncd -- %s\nexec %s -m harness.runner --run %s\n" % (
        shlex.quote(str(PROJECT_ROOT)), shlex.quote(sys.executable), shlex.quote(run_id))
    script.write_text(body, encoding="utf-8")
    argv = ["sbatch", "--parsable", "--nodes", "1", "--ntasks", "1",
            "--job-name", "fmca-" + run_id[-80:],
            "--output", str(run_dir / "stdout.log"), "--error", str(run_dir / "stderr.log")]
    argv.extend(str(item) for item in config.get("slurm_args", []))
    argv.extend(["--partition", ",".join(selected_partitions(config, gpus, profile))])
    if gpus:
        argv.extend(["--gres", "gpu:%d" % gpus])
    argv.append(str(script))
    result = subprocess.run(argv, cwd=str(PROJECT_ROOT), text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        update_job(run_id, {"state": "BLOCKED", "end_time": now_iso(), "exit_code": result.returncode,
                            "failure_reason": "sbatch failed: " + result.stderr.strip()}, config)
        raise RuntimeError("sbatch failed: " + result.stderr.strip())
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id:
        update_job(run_id, {"state": "BLOCKED", "end_time": now_iso(),
                            "failure_reason": "sbatch returned no job ID"}, config)
        raise RuntimeError("sbatch returned no job ID")
    update_job(run_id, {"slurm_job_id": job_id}, config)


def submit_new(name: str, gpus: int, command: Sequence[str], retry_from: Optional[str] = None,
               profile: str = "default") -> Tuple[int, Optional[str]]:
    run_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    try:
        config = load_config()
        mode = scheduler.resolve_mode(config)
        raw = normalize_command(command)
        final = build_command(raw, gpus)
        with locked():
            jobs = refresh_all_locked(config)
            allocated, _used = allocation_for(config, mode, gpus, jobs)
            run_id, run_dir = _create_run_locked(name, gpus, raw, final, mode, allocated, config,
                                                 retry_from, profile)
            if mode == "direct":
                _launch_direct(run_id, run_dir, config)
            else:
                _launch_slurm(run_id, run_dir, gpus, config, profile)
        print(run_id)
        return 0, run_id
    except Exception as exc:
        if run_id is not None and config is not None:
            try:
                with locked():
                    current = load_jobs()["jobs"].get(run_id, {})
                    if current.get("state") in ACTIVE_STATES:
                        update_job(run_id, {"state": "BLOCKED", "end_time": now_iso(),
                                            "failure_reason": "launch failed: %s" % exc}, config)
            except Exception:
                pass
        return fail(str(exc), 3), None


def submit(args: argparse.Namespace) -> int:
    code, _ = submit_new(args.name, args.gpus, args.command, profile=args.profile)
    return code


def _duration(record: Dict[str, Any]) -> str:
    start = record.get("start_time") or record.get("created_at")
    end = record.get("end_time") or now_iso()
    if not start:
        return ""
    try:
        seconds = max(0, int((dt.datetime.fromisoformat(end) - dt.datetime.fromisoformat(start)).total_seconds()))
    except (TypeError, ValueError):
        return ""
    return "%02d:%02d:%02d" % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def status(args: argparse.Namespace) -> int:
    try:
        config = load_config()
        with locked():
            jobs = refresh_all_locked(config)
            records = jobs["jobs"]
            if args.run:
                if args.run not in records:
                    return fail("unknown run: " + args.run)
                records = {args.run: records[args.run]}
    except Exception as exc:
        return fail(str(exc))
    print("RUN_ID\tSTATE\tGPUS\tGPU_IDS\tDURATION\tEXIT")
    for run_id, record in sorted(records.items()):
        print("%s\t%s\t%s\t%s\t%s\t%s" % (
            run_id, record.get("state", ""), record.get("requested_gpus", ""),
            ",".join(str(item) for item in record.get("actual_gpu_ids", [])), _duration(record),
            "" if record.get("exit_code") is None else record.get("exit_code")))
    return 0


def stop(args: argparse.Namespace) -> int:
    try:
        config = load_config()
        grace = int(config.get("stop_grace_seconds", 30))
        with locked():
            jobs = refresh_all_locked(config)
            if args.run not in jobs["jobs"]:
                return fail("unknown run: " + args.run)
            record = jobs["jobs"][args.run]
            if record.get("state") not in ACTIVE_STATES:
                return fail("run is not active: %s (%s)" % (args.run, record.get("state")))
            mode = record.get("resolved_mode")
            pid = record.get("pid")
            job_id = record.get("slurm_job_id")
            if mode == "slurm":
                if not job_id:
                    return fail("run has no Slurm job ID")
                result = subprocess.run(["scancel", str(job_id)], text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, check=False)
                if result.returncode != 0:
                    return fail("scancel failed: " + result.stderr.strip(), 3)
                update_job(args.run, {"state": "STOPPED", "end_time": now_iso(),
                                      "failure_reason": "stopped by operator"}, config)
                print(args.run + " STOPPED")
                return 0
            if not pid or not pid_is_harness_run(pid, args.run):
                update_job(args.run, {"state": "FAILED", "end_time": now_iso(),
                                      "failure_reason": "process disappeared before stop"}, config)
                return fail("process is not alive; stale state repaired", 3)
            target_pid = int(pid)
            os.killpg(target_pid, signal.SIGTERM)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and pid_alive(target_pid):
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        if pid_alive(target_pid):
            os.killpg(target_pid, signal.SIGKILL)
        with locked():
            update_job(args.run, {"state": "STOPPED", "end_time": now_iso(),
                                  "failure_reason": "stopped by operator"}, config)
        print(args.run + " STOPPED")
        return 0
    except ProcessLookupError:
        with locked():
            update_job(args.run, {"state": "STOPPED", "end_time": now_iso(),
                                  "failure_reason": "stopped by operator"}, config)
        print(args.run + " STOPPED")
        return 0
    except Exception as exc:
        return fail(str(exc), 3)


def retry(args: argparse.Namespace) -> int:
    try:
        config = load_config()
        with locked():
            jobs = refresh_all_locked(config)
            if args.run not in jobs["jobs"]:
                return fail("unknown run: " + args.run)
            original_record = jobs["jobs"][args.run]
            if original_record.get("state") in ACTIVE_STATES:
                return fail("cannot retry an active run")
            request = read_json(runs_dir(config) / args.run / "request.json")
        code, _ = submit_new(request["name"], int(request["requested_gpus"]),
                             request["original_command"], retry_from=args.run,
                             profile=request.get("profile", "default"))
        return code
    except Exception as exc:
        return fail(str(exc), 3)


def _last_metric(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    last: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    last = value
            except json.JSONDecodeError:
                continue
    return last


def collect(args: argparse.Namespace) -> int:
    try:
        config = load_config()
        with locked():
            jobs = refresh_all_locked(config)
            records = jobs["jobs"]
            if args.run:
                if args.run not in records:
                    return fail("unknown run: " + args.run)
                records = {args.run: records[args.run]}
        rows = []
        metric_keys = set()
        for run_id, record in sorted(records.items()):
            if record.get("state") not in FINAL_STATES:
                continue
            metric = _last_metric(runs_dir(config) / run_id / "metrics.jsonl")
            metric_keys.update(str(key) for key in metric)
            rows.append((record, metric))
        base_fields = ["run_id", "name", "state", "requested_gpus", "actual_gpu_ids", "created_at",
                       "start_time", "end_time", "duration", "exit_code", "failure_reason", "retry_from"]
        fields = base_fields + ["metric_" + key for key in sorted(metric_keys)]
        destination = runs_dir(config) / "summary.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record, metric in rows:
                row = {key: record.get(key, "") for key in base_fields}
                row["actual_gpu_ids"] = ",".join(str(item) for item in record.get("actual_gpu_ids", []))
                row["duration"] = _duration(record)
                for key, value in metric.items():
                    row["metric_" + str(key)] = value
                writer.writerow(row)
        os.replace(str(temporary), str(destination))
        print(str(destination))
        return 0
    except Exception as exc:
        return fail(str(exc), 3)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FMCA-AV file-based experiment harness")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("doctor", help="report environment without changing it").set_defaults(func=doctor)
    for action, function in (("dry-run", dry_run), ("submit", submit)):
        child = subparsers.add_parser(action)
        child.add_argument("--name", required=True)
        child.add_argument("--gpus", required=True, type=validate_gpus)
        child.add_argument("--profile", choices=("default", "imagenet"), default="default",
                           help="GPU partition policy; imagenet excludes V100")
        child.add_argument("command", nargs=argparse.REMAINDER)
        child.set_defaults(func=function)
    child = subparsers.add_parser("status")
    child.add_argument("--run")
    child.set_defaults(func=status)
    child = subparsers.add_parser("stop")
    child.add_argument("--run", required=True)
    child.set_defaults(func=stop)
    child = subparsers.add_parser("retry")
    child.add_argument("--run", required=True)
    child.set_defaults(func=retry)
    child = subparsers.add_parser("collect")
    child.add_argument("--run")
    child.set_defaults(func=collect)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
