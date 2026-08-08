#!/usr/bin/env python3
"""Launch low-label linear/fine-tune controls for four key SSL baselines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CROSS_WATCHER = "20260807-060445_launch-cross-dataset-baseline-wave-fixed"
METHODS = ("simclr", "vicreg", "moco_v2", "dino")
C10_RUNS = {
    "simclr": "20260807-053905_cifar10-simclr-5epoch-screening",
    "vicreg": "20260807-055327_cifar10-vicreg-5epoch-screening",
    "moco_v2": "20260807-060309_cifar10-moco_v2-5epoch-screening",
    "dino": "20260807-060716_cifar10-dino-5epoch-screening",
}
DATASETS = {
    "cifar10": "configs/ssl/cifar10_baseline_smoke.json",
    "cifar100": "configs/ssl/cifar100_smoke.json",
    "imagenet100": "configs/ssl/imagenet100_smoke.json",
}


def refresh() -> None: subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def state(run_id: str) -> str: return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def latest_retry(run_id: str) -> str:
    jobs = dict(json.loads(Path("harness/state/jobs.json").read_text(encoding="utf-8")).get("jobs", {}))
    current = run_id; visited = {current}
    while True:
        children = [value for value in jobs.values() if str(value.get("retry_from", "")) == current
                    and str(value.get("run_id", "")) not in visited]
        if not children: return current
        chosen = max(children, key=lambda value: str(value.get("created_at", "")))
        current = str(chosen["run_id"]); visited.add(current)


def wait_success(run_id: str) -> None:
    while True:
        refresh(); run_id = latest_retry(run_id); value = state(run_id)
        if value == "SUCCEEDED": return
        if value in {"FAILED", "STOPPED", "BLOCKED"}: raise RuntimeError(f"{run_id} ended in {value}")
        time.sleep(POLL_SECONDS)


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def cross_sources(watcher: str) -> dict[tuple[str, str], str]:
    values = {}
    path = Path("runs") / watcher / "artifacts" / "submitted_jobs.tsv"
    for line in path.read_text(encoding="utf-8").splitlines():
        dataset, method, run_id = line.split("\t")
        values[(dataset, method)] = run_id
    return values


def checkpoint(run_id: str) -> str:
    result = json.loads((Path("runs") / run_id / "artifacts" / "train_result.json").read_text(encoding="utf-8"))
    value = result.get("best_checkpoint") or result.get("last_checkpoint")
    if not value: raise RuntimeError(f"checkpoint missing for {run_id}")
    return str(value)


def save(records: list[dict[str, object]], output: Path) -> None:
    temporary = output.with_suffix(".tmp"); temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("all", "non-imagenet", "imagenet"),
        default="all",
        help="Select a scheduling phase so ImageNet work can be deferred.",
    )
    args = parser.parse_args()
    artifacts = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True); output = artifacts / "submitted.json"
    time.sleep(POLL_SECONDS); wait_success(CROSS_WATCHER)
    cross_watcher = latest_retry(CROSS_WATCHER); sources = cross_sources(cross_watcher)
    records = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else []
    refresh()
    for record in records:
        run_id = str(record["run_id"])
        if state(run_id) in {"FAILED", "STOPPED", "BLOCKED"}:
            record["run_id"] = submit(["python3", "-m", "harness.cli", "retry", "--run", run_id])
            save(records, output)
    existing = {(str(record["dataset"]), str(record["method"]), str(record["protocol"]),
                 float(record["fraction"])) for record in records}
    selected_datasets = {
        dataset: config for dataset, config in DATASETS.items()
        if (args.phase == "all"
            or (args.phase == "imagenet" and dataset == "imagenet100")
            or (args.phase == "non-imagenet" and dataset != "imagenet100"))
    }
    for dataset, config in selected_datasets.items():
        profile = ["--profile", "imagenet"] if dataset == "imagenet100" else []
        for method_index, method in enumerate(METHODS):
            source_run = C10_RUNS[method] if dataset == "cifar10" else sources[(dataset, method)]
            wait_success(source_run); source_checkpoint = checkpoint(source_run)
            for protocol, fractions in (("linear-probe", (0.01, 0.1, 1.0)), ("fine-tune", (0.01, 0.1))):
                for fraction in fractions:
                    if (dataset, method, protocol, fraction) in existing:
                        continue
                    tag = str(fraction).replace(".", "p"); seed = 20276000 + len(records)
                    override = {"experiment": {"method": method}, "probe": {"label_fraction": fraction}}
                    name = f"e6-{dataset}-{method}-{protocol}-{tag}"
                    run_id = submit([
                        "python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", *profile, "--",
                        "env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":")), f"FMCA_SEED_OVERRIDE={seed}",
                        PYTHON, "-m", "fmca_av.baseline_cli", protocol, "--config", config, "--checkpoint", source_checkpoint,
                    ])
                    records.append({"dataset": dataset, "method": method, "protocol": protocol, "fraction": fraction, "source_run": source_run, "run_id": run_id})
                    existing.add((dataset, method, protocol, fraction))
                    save(records, output)
    wait_successful = [str(record["run_id"]) for record in records]
    while True:
        time.sleep(POLL_SECONDS); refresh(); values = {run_id: state(run_id) for run_id in wait_successful}
        failures = {run_id: value for run_id, value in values.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failures: raise RuntimeError("baseline low-label children failed: " + json.dumps(failures, sort_keys=True))
        if all(value == "SUCCEEDED" for value in values.values()): break
    return 0


if __name__ == "__main__": raise SystemExit(main())
