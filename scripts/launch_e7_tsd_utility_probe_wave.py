#!/usr/bin/env python3
"""Attach frozen linear-probe utility to every preregistered TSD sweep run."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
C10_WATCHER = "20260807-060445_launch-cifar10-tsd-full-severity-sweep-fixed"
CROSS_WATCHER = "20260807-071234_launch-e7-crossscale-tsd-sweep"
CONFIGS = {"cifar10": "configs/ssl/cifar10_smoke.json", "cifar100": "configs/ssl/cifar100_smoke.json", "imagenet100": "configs/ssl/imagenet100_smoke.json"}


def refresh() -> None: subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)
def state(run_id: str) -> str: return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def wait_all(run_ids: list[str]) -> None:
    while True:
        refresh(); values = {run_id: state(run_id) for run_id in run_ids}; failed = {run_id: value for run_id, value in values.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failed: raise RuntimeError("TSD utility prerequisite failed: " + json.dumps(failed, sort_keys=True))
        if all(value == "SUCCEEDED" for value in values.values()): return
        time.sleep(POLL_SECONDS)


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def save(records: list[dict[str, object]], output: Path) -> None:
    temporary = output.with_suffix(".tmp"); temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)


def checkpoint(run_id: str) -> str:
    result = json.loads((Path("runs") / run_id / "artifacts" / "train_result.json").read_text(encoding="utf-8")); value = result.get("best_checkpoint") or result.get("last_checkpoint")
    if not value: raise RuntimeError(f"checkpoint missing for {run_id}")
    return str(value)


def main() -> int:
    artifacts = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True); output = artifacts / "submitted.json"
    time.sleep(POLL_SECONDS); wait_all([C10_WATCHER, CROSS_WATCHER])
    source_records = []
    for watcher, default_dataset in ((C10_WATCHER, "cifar10"), (CROSS_WATCHER, "")):
        values = json.loads((Path("runs") / watcher / "artifacts" / "submitted.json").read_text(encoding="utf-8"))
        for value in values:
            source_records.append({**value, "dataset": value.get("dataset", default_dataset)})
    wait_all([str(record["run_id"]) for record in source_records]); submitted = []
    for source in source_records:
        dataset = str(source["dataset"]); profile = ["--profile", "imagenet"] if dataset == "imagenet100" else []
        name = f"{source['run_id']}-utility-linear-probe"; override = {"data": {"augmentation": source["overrides"]}}
        run_id = submit([
            "python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", *profile, "--",
            "env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":")), f"FMCA_SEED_OVERRIDE={source['seed']}",
            PYTHON, "-m", "fmca_av.cli", "linear-probe", "--config", CONFIGS[dataset], "--checkpoint", checkpoint(str(source["run_id"])),
        ])
        submitted.append({"dataset": dataset, "channel": source["channel"], "level": source["level"], "seed": source["seed"], "source_run": source["run_id"], "probe_run": run_id}); save(submitted, output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
