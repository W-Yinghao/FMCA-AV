#!/usr/bin/env python3
"""Train a map-compatible VGG reference and launch E9 localization controls."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG = "configs/ssl/imagenet100_smoke.json"
OVERRIDE = {"model": {"backbone": "vgg16_bn"}, "data": {"batch_size": 32}}
DATASETS = {
    "cub": ("/projects/EEG-foundation-model/yinghao/FMCA-AV/cub", []),
    "voc": ("/projects/EEG-foundation-model/yinghao/FMCA-AV/voc/VOC2012", []),
    "imagenet": ("/projects/EEG-foundation-model/yinghao/FMCA-AV/imagenet/ILSVRC", ["--labels", "/projects/common/imagenet/LOC_val_solution.csv"]),
}


def refresh() -> None: subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)
def state(run_id: str) -> str: return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def wait_success(run_id: str) -> None:
    while True:
        refresh(); value = state(run_id)
        if value == "SUCCEEDED": return
        if value in {"FAILED", "STOPPED", "BLOCKED"}: raise RuntimeError(f"{run_id} ended in {value}")
        time.sleep(POLL_SECONDS)


def save(records: list[dict[str, object]], output: Path) -> None:
    temporary = output.with_suffix(".tmp"); temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)


def main() -> int:
    artifacts = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True); output = artifacts / "submitted.json"; records = []
    time.sleep(POLL_SECONDS); environment = json.dumps(OVERRIDE, separators=(",", ":"))
    train_run = submit(["python3", "-m", "harness.cli", "submit", "--name", "imagenet100-vgg16-bn-32step-smoke", "--gpus", "1", "--profile", "imagenet", "--", "env", "FMCA_CONFIG_OVERRIDES=" + environment, "bash", "scripts/run_fmca_pipeline.sh", "--config", CONFIG])
    records.append({"stage": "train", "run_id": train_run}); save(records, output); wait_success(train_run)
    result = json.loads((Path("runs") / train_run / "artifacts" / "train_result.json").read_text(encoding="utf-8")); checkpoint = result.get("best_checkpoint") or result.get("last_checkpoint"); calibration = Path("runs") / train_run / "artifacts" / "calibration.pt"
    if not checkpoint: raise RuntimeError("VGG checkpoint missing")
    for dataset, (root, extra) in DATASETS.items():
        run_id = submit(["python3", "-m", "harness.cli", "submit", "--name", f"imagenet100-vgg16-bn-{dataset}-localization", "--gpus", "1", "--profile", "imagenet", "--", "env", "FMCA_CONFIG_OVERRIDES=" + environment, PYTHON, "-m", "scripts.run_dependence_localization", "--config", CONFIG, "--checkpoint", str(checkpoint), "--calibration", str(calibration), "--dataset", dataset, "--root", root, "--samples", "100", *extra])
        records.append({"stage": "localization", "dataset": dataset, "randomized": False, "run_id": run_id}); save(records, output)
    run_id = submit(["python3", "-m", "harness.cli", "submit", "--name", "imagenet100-vgg16-bn-randomized-cub-localization", "--gpus", "1", "--profile", "imagenet", "--", "env", "FMCA_CONFIG_OVERRIDES=" + environment, PYTHON, "-m", "scripts.run_dependence_localization", "--config", CONFIG, "--checkpoint", str(checkpoint), "--calibration", str(calibration), "--dataset", "cub", "--root", DATASETS["cub"][0], "--samples", "100", "--randomize-backbone"])
    records.append({"stage": "localization", "dataset": "cub", "randomized": True, "run_id": run_id}); save(records, output); return 0


if __name__ == "__main__": raise SystemExit(main())
