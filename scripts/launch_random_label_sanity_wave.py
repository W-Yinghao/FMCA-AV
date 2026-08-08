#!/usr/bin/env python3
"""Train a fixed-random-label control and evaluate its CUB maps."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG = "configs/ssl/imagenet100_smoke.json"
OVERRIDE = {"experiment": {"name": "imagenet100-supervised-random-labels", "random_labels": True}}


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def state(run_id: str) -> str:
    return str(read(Path("runs") / run_id / "status.json")["state"])


def wait_success(run_id: str) -> None:
    while True:
        time.sleep(POLL_SECONDS); refresh(); value = state(run_id)
        if value == "SUCCEEDED": return
        if value in {"FAILED", "STOPPED", "BLOCKED"}: raise RuntimeError(f"random-label run {run_id} ended in {value}")


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def main() -> int:
    artifacts = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True)
    time.sleep(POLL_SECONDS)
    environment = "FMCA_CONFIG_OVERRIDES=" + json.dumps(OVERRIDE, separators=(",", ":"))
    train_run = submit([
        "python3", "-m", "harness.cli", "submit", "--name", "imagenet100-supervised-random-labels-32step",
        "--gpus", "1", "--profile", "imagenet", "--", "env", environment,
        PYTHON, "-m", "fmca_av.supervised_cli", "--config", CONFIG,
    ])
    (artifacts / "train_run.txt").write_text(train_run + "\n", encoding="utf-8")
    wait_success(train_run)
    result = read(Path("runs") / train_run / "artifacts" / "supervised_result.json")
    checkpoint = result.get("best_checkpoint") or result.get("last_checkpoint")
    localization_run = submit([
        "python3", "-m", "harness.cli", "submit", "--name", "imagenet100-supervised-random-labels-cub-localization",
        "--gpus", "1", "--profile", "imagenet", "--", "env", environment,
        PYTHON, "-m", "scripts.run_supervised_localization", "--config", CONFIG,
        "--checkpoint", str(checkpoint), "--dataset", "cub",
        "--root", "/projects/EEG-foundation-model/yinghao/FMCA-AV/cub", "--samples", "100",
    ])
    (artifacts / "localization_run.txt").write_text(localization_run + "\n", encoding="utf-8")
    wait_success(localization_run)
    composition_run = submit([
        "python3", "-m", "harness.cli", "submit", "--name", "imagenet100-supervised-random-labels-cnn-composition",
        "--gpus", "1", "--profile", "imagenet", "--", "env", environment,
        PYTHON, "-m", "scripts.run_cnn_composition_maps", "--config", CONFIG,
        "--checkpoint", str(checkpoint), "--model-type", "supervised",
        "--root", "/projects/EEG-foundation-model/yinghao/FMCA-AV/cub",
        "--calibration-samples", "50", "--evaluation-samples", "50",
    ])
    (artifacts / "composition_run.txt").write_text(composition_run + "\n", encoding="utf-8")
    wait_success(composition_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
