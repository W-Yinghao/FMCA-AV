#!/usr/bin/env python3
"""Launch the frozen CIFAR-10 V=2/V=8 baseline screening matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Optional


POLL_SECONDS = 300
PREREQUISITE = "20260807-060401_continue-cifar10-baseline-screening-fixed"
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG = "configs/ssl/cifar10_baseline_smoke.json"
METHODS = ("simclr", "barlow_twins", "vicreg", "spectral_contrastive", "fastsiam", "byol", "moco_v2", "dino")
SEEDS = (20275001, 20275002, 20275003)
PROBE_BATCH_LIMIT = 3


def refresh(run_id: Optional[str] = None) -> None:
    command = ["python3", "-m", "harness.cli", "status"]
    if run_id is not None:
        command += ["--run", run_id]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def state(run_id: str) -> str:
    return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def wait_all(run_ids: list[str]) -> None:
    while True:
        refresh()
        states = {run_id: state(run_id) for run_id in run_ids}
        failures = {run_id: value for run_id, value in states.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failures:
            raise RuntimeError("matched-view prerequisite failed: " + json.dumps(failures, sort_keys=True))
        if all(value == "SUCCEEDED" for value in states.values()):
            return
        time.sleep(POLL_SECONDS)


def submit(command: list[str]) -> str:
    while True:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def checkpoint(run_id: str) -> str:
    result = json.loads((Path("runs") / run_id / "artifacts" / "train_result.json").read_text(encoding="utf-8"))
    value = result.get("best_checkpoint") or result.get("last_checkpoint")
    if not value:
        raise RuntimeError(f"no checkpoint recorded for {run_id}")
    return str(value)


def save(records: list[dict[str, object]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def repair_failed(records: list[dict[str, object]], key: str, output: Path) -> None:
    refresh()
    for record in records:
        value = record.get(key)
        if not value or state(str(value)) not in {"FAILED", "STOPPED", "BLOCKED"}:
            continue
        record[key] = submit([
            "python3", "-m", "harness.cli", "retry", "--run", str(value),
        ])
        save(records, output)


def main() -> int:
    run_dir = Path(os.environ["FMCA_HARNESS_RUN_DIR"]); artifacts = run_dir / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True)
    time.sleep(POLL_SECONDS); wait_all([PREREQUISITE])
    output = artifacts / "submitted.json"
    records: list[dict[str, object]] = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else []
    existing = {(str(record["method"]), int(record["views"]), int(record["seed_index"])) for record in records}
    for views in (2, 8):
        for method in METHODS:
            for seed_index, seed in enumerate(SEEDS, 1):
                if (method, views, seed_index) in existing:
                    continue
                name = f"e5-cifar10-{method}-v{views}-seed{seed_index}-screening"
                override = {"experiment": {"name": name, "method": method}, "data": {"num_views": views}}
                run_id = submit([
                    "python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", "--",
                    "env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":")),
                    f"FMCA_SEED_OVERRIDE={seed}", PYTHON, "-m", "fmca_av.baseline_cli", "train", "--config", CONFIG,
                ])
                records.append({"method": method, "views": views, "seed_index": seed_index, "seed": seed, "train_run": run_id})
                existing.add((method, views, seed_index))
                save(records, output)
    time.sleep(POLL_SECONDS)
    repair_failed(records, "train_run", output)
    wait_all([str(record["train_run"]) for record in records])
    repair_failed(records, "probe_run", output)
    probe_batch: list[str] = []
    for record in records:
        if record.get("probe_run"):
            continue
        if len(probe_batch) >= PROBE_BATCH_LIMIT:
            wait_all(probe_batch)
            probe_batch = []
        method = str(record["method"]); views = int(record["views"]); seed_index = int(record["seed_index"]); seed = int(record["seed"])
        name = f"e5-cifar10-{method}-v{views}-seed{seed_index}-linear-probe"
        override = {"experiment": {"method": method}, "data": {"num_views": views}}
        probe_run = submit([
            "python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", "--",
            "env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":")), f"FMCA_SEED_OVERRIDE={seed}",
            PYTHON, "-m", "fmca_av.baseline_cli", "linear-probe", "--config", CONFIG,
            "--checkpoint", checkpoint(str(record["train_run"])),
        ])
        record["probe_run"] = probe_run; probe_batch.append(probe_run); save(records, output)
    if probe_batch:
        wait_all(probe_batch)
    time.sleep(POLL_SECONDS)
    repair_failed(records, "probe_run", output)
    wait_all([str(record["probe_run"]) for record in records])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
