#!/usr/bin/env python3
"""Probe TSD sweep checkpoints after corrected CIFAR-10 and cross-scale waves."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIGS = {
    "cifar10": "configs/ssl/cifar10_smoke.json",
    "cifar100": "configs/ssl/cifar100_smoke.json",
    "imagenet100": "configs/ssl/imagenet100_smoke.json",
}


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def state(run_id: str) -> str:
    return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def wait_all(run_ids: list[str]) -> None:
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        states = {run_id: state(run_id) for run_id in run_ids}
        failed = {run_id: value for run_id, value in states.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failed:
            raise RuntimeError("TSD utility prerequisite failed: " + json.dumps(failed, sort_keys=True))
        if all(value == "SUCCEEDED" for value in states.values()):
            return


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def save(records: list[dict[str, object]], output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def checkpoint(run_id: str) -> str:
    payload = json.loads((Path("runs") / run_id / "artifacts" / "train_result.json").read_text(encoding="utf-8"))
    value = payload.get("best_checkpoint") or payload.get("last_checkpoint")
    if not value:
        raise RuntimeError(f"checkpoint missing for {run_id}")
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c10-watcher", required=True)
    parser.add_argument("--cross-watcher", required=True)
    args = parser.parse_args()
    artifacts = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    output = artifacts / "submitted.json"
    wait_all([args.c10_watcher, args.cross_watcher])
    sources: list[dict[str, object]] = []
    for watcher, default_dataset in ((args.c10_watcher, "cifar10"), (args.cross_watcher, "")):
        values = json.loads((Path("runs") / watcher / "artifacts" / "submitted.json").read_text(encoding="utf-8"))
        for value in values:
            sources.append({**value, "dataset": value.get("dataset", default_dataset)})
    wait_all([str(source["run_id"]) for source in sources])
    records: list[dict[str, object]] = []
    for source in sources:
        dataset = str(source["dataset"])
        profile = ["--profile", "imagenet"] if dataset == "imagenet100" else []
        run_id = submit([
            "python3", "-m", "harness.cli", "submit", "--name", f"{source['run_id']}-utility-linear-probe",
            "--gpus", "1", *profile, "--", "env", f"FMCA_SEED_OVERRIDE={source['seed']}",
            PYTHON, "-m", "fmca_av.cli", "linear-probe",
            "--config", CONFIGS[dataset], "--checkpoint", checkpoint(str(source["run_id"])),
        ])
        records.append({
            "dataset": dataset, "channel": source["channel"], "level": source["level"],
            "seed": source["seed"], "source_run": source["run_id"], "probe_run": run_id,
        })
        save(records, output)
    wait_all([str(record["probe_run"]) for record in records])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
