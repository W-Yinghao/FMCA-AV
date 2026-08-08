#!/usr/bin/env python3
"""Summarize GPU, batch-timing, and vmstat samples from a profiled run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "maximum": max(values) if values else None,
    }


def gpu_summary(path: Path, start_time: str | None = None,
                end_time: str | None = None) -> dict[str, object]:
    columns: dict[str, list[float]] = {
        "gpu_util_pct": [], "memory_util_pct": [], "memory_used_mib": [],
        "memory_total_mib": [], "power_w": [], "sm_clock_mhz": [],
    }
    names: set[str] = set()
    devices: dict[str, dict[str, list[float]]] = {}
    start = datetime.fromisoformat(start_time) if start_time else None
    end = datetime.fromisoformat(end_time) if end_time else None
    if not path.is_file():
        return {"sample_count": 0}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                sampled = datetime.fromisoformat(str(row.get("sample_time", "")))
            except ValueError:
                sampled = None
            if sampled is not None and ((start is not None and sampled < start) or
                                        (end is not None and sampled > end)):
                continue
            names.add(str(row.get("name", "")).strip())
            device_id = str(row.get("index", "unknown")).strip()
            device_columns = devices.setdefault(device_id, {key: [] for key in columns})
            for key in columns:
                try:
                    value = float(str(row.get(key, "")).strip())
                    columns[key].append(value)
                    device_columns[key].append(value)
                except ValueError:
                    pass
    result: dict[str, object] = {key: stats(values) for key, values in columns.items()}
    result["sample_count"] = len(columns["gpu_util_pct"])
    result["gpu_names"] = sorted(name for name in names if name)
    result["per_gpu"] = {
        device_id: {key: stats(values) for key, values in device_columns.items()}
        for device_id, device_columns in sorted(devices.items())
    }
    result["sampling_window"] = {"start_time": start_time, "end_time": end_time}
    totals = columns["memory_total_mib"]
    used = columns["memory_used_mib"]
    result["peak_memory_fraction"] = max(used) / max(totals) if used and totals and max(totals) else None
    return result


def sampling_bounds(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    first: datetime | None = None
    last: datetime | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                sampled = datetime.fromisoformat(str(row.get("sample_time", "")))
            except ValueError:
                continue
            first = sampled if first is None else first
            last = sampled
    return {
        "sampler_start_time": first.isoformat() if first else None,
        "sampler_end_time": last.isoformat() if last else None,
    }


def vmstat_summary(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"sample_count": 0}
    names: tuple[str, ...] = ()
    columns: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if fields[:2] == ["r", "b"] and "id" in fields:
            detected = tuple(fields)
            if not names:
                names = detected
                columns = {name: [] for name in names}
            elif detected != names:
                raise ValueError(
                    "vmstat columns changed within one log: %s -> %s" % (names, detected)
                )
            continue
        if not names:
            continue
        if len(fields) != len(names) or not all(field.lstrip("-").isdigit() for field in fields):
            continue
        for name, value in zip(names, fields):
            columns[name].append(float(value))
    if not columns:
        return {"sample_count": 0}
    result: dict[str, object] = {key: stats(values) for key, values in columns.items()}
    result["sample_count"] = len(columns["id"])
    return result


def resource_usage(path: Path) -> dict[str, object]:
    """Parse the stable key/value portion of GNU time -v output."""
    if not path.is_file():
        return {}
    result: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ": " not in line:
            continue
        key, value = line.strip().rsplit(": ", 1)
        normalized = key.lower().replace(" ", "_").replace("(kbytes)", "kb")
        cleaned = value.strip()
        try:
            result[normalized] = int(cleaned)
        except ValueError:
            try:
                result[normalized] = float(cleaned.rstrip("%"))
            except ValueError:
                result[normalized] = cleaned
    return result


def classify(gpu: dict[str, object], batch: dict[str, object], vmstat: dict[str, object],
             stderr: str) -> list[str]:
    findings: list[str] = []
    if "out of memory" in stderr.lower():
        findings.append("GPU_MEMORY_OOM")
    memory_fraction = gpu.get("peak_memory_fraction")
    if isinstance(memory_fraction, (int, float)) and memory_fraction >= 0.90:
        findings.append("GPU_MEMORY_PRESSURE")
    util = dict(gpu.get("gpu_util_pct", {})).get("mean") if gpu.get("gpu_util_pct") else None
    wait_fraction = batch.get("inter_batch_wait_fraction")
    compute_fraction = batch.get("compute_fraction")
    io_wait = dict(vmstat.get("wa", {})).get("mean") if vmstat.get("wa") else None
    cpu_idle = dict(vmstat.get("id", {})).get("mean") if vmstat.get("id") else None
    if isinstance(util, (int, float)) and util >= 90 and isinstance(compute_fraction, (int, float)) and compute_fraction >= 0.90:
        findings.append("GPU_COMPUTE_BOUND")
    if isinstance(wait_fraction, (int, float)) and wait_fraction >= 0.25 and isinstance(util, (int, float)) and util < 80:
        findings.append("INPUT_PIPELINE_WAIT")
    elif isinstance(wait_fraction, (int, float)) and wait_fraction >= 0.15:
        findings.append("INPUT_PIPELINE_PARTIAL_WAIT")
    if isinstance(io_wait, (int, float)) and io_wait >= 10:
        findings.append("STORAGE_IO_WAIT")
    if isinstance(cpu_idle, (int, float)) and cpu_idle <= 10 and isinstance(util, (int, float)) and util < 80:
        findings.append("CPU_SATURATION")
    if not findings and gpu.get("sample_count", 0):
        findings.append("MIXED_OR_INSUFFICIENT_EVIDENCE")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    run_dir = Path("runs") / args.run
    artifacts = run_dir / "artifacts"
    batch_path = artifacts / "batch_profile.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8")) if batch_path.is_file() else {}
    bounds = sampling_bounds(artifacts / "gpu_samples.csv")
    gpu = gpu_summary(
        artifacts / "gpu_samples.csv",
        str(batch.get("train_start_time")) if batch.get("train_start_time") else None,
        str(batch.get("last_update_time")) if batch.get("last_update_time") else None,
    )
    vmstat = vmstat_summary(artifacts / "vmstat.log")
    resources = resource_usage(artifacts / "resource_usage.txt")
    stderr_path = run_dir / "stderr.log"
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    train_start = datetime.fromisoformat(str(batch["train_start_time"])) if batch.get("train_start_time") else None
    sampler_start = (
        datetime.fromisoformat(str(bounds["sampler_start_time"]))
        if bounds.get("sampler_start_time") else None
    )
    startup = {
        **bounds,
        "train_start_time": train_start.isoformat() if train_start else None,
        "seconds_before_training": (
            (train_start - sampler_start).total_seconds()
            if train_start is not None and sampler_start is not None else None
        ),
    }
    payload = {
        "run_id": args.run,
        "state": status.get("state"),
        "gpu": gpu,
        "batch_timing": batch,
        "vmstat": vmstat,
        "resource_usage": resources,
        "startup_timing": startup,
        "findings": classify(gpu, batch, vmstat, stderr),
    }
    destination = Path(args.output) if args.output else artifacts / "profile_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
