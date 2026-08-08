#!/usr/bin/env python3
"""Rerun ResNet/ConvNeXt/ViT E9 maps with deletion-insertion AUC."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
SOURCES = {
    "resnet50": ("20260807-052202_imagenet1k-lightning-32step-smoke", "configs/ssl/imagenet1k_smoke.json", {}),
    "convnext_tiny": ("20260807-060227_imagenet100-convnext_tiny-32step-smoke", "configs/ssl/imagenet100_smoke.json", {"model": {"backbone": "convnext_tiny"}, "data": {"batch_size": 32}}),
    "vit_s_16": ("20260807-060729_imagenet100-vit_s_16-32step-smoke", "configs/ssl/imagenet100_smoke.json", {"model": {"backbone": "vit_s_16"}, "data": {"batch_size": 32}}),
}
DATASETS = {
    "cub": ("/projects/EEG-foundation-model/yinghao/FMCA-AV/cub", []),
    "voc": ("/projects/EEG-foundation-model/yinghao/FMCA-AV/voc/VOC2012", []),
    "imagenet": ("/projects/EEG-foundation-model/yinghao/FMCA-AV/imagenet/ILSVRC", ["--labels", "/projects/common/imagenet/LOC_val_solution.csv"]),
}


def refresh() -> None: subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def save(records: list[dict[str, object]], output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)


def main() -> int:
    artifacts = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True)
    output = artifacts / "submitted.json"; records: list[dict[str, object]] = []; time.sleep(POLL_SECONDS)
    for architecture, (source_run, config, override) in SOURCES.items():
        result = json.loads((Path("runs") / source_run / "artifacts" / "train_result.json").read_text(encoding="utf-8"))
        checkpoint = result.get("best_checkpoint") or result.get("last_checkpoint")
        if not checkpoint: raise RuntimeError(f"checkpoint missing for {source_run}")
        calibration = Path("runs") / source_run / "artifacts" / "calibration.pt"
        environment = ["env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":"))] if override else []
        for dataset, (root, extra) in DATASETS.items():
            name = f"e9-v2-{architecture}-{dataset}-faithfulness-auc"
            run_id = submit(["python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", "--profile", "imagenet", "--",
                *environment, PYTHON, "-m", "scripts.run_dependence_localization", "--config", config,
                "--checkpoint", str(checkpoint), "--calibration", str(calibration), "--dataset", dataset,
                "--root", root, "--samples", "100", *extra])
            records.append({"architecture": architecture, "dataset": dataset, "randomized": False, "source_run": source_run, "run_id": run_id}); save(records, output)
        name = f"e9-v2-{architecture}-randomized-cub-faithfulness-auc"
        run_id = submit(["python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", "--profile", "imagenet", "--",
            *environment, PYTHON, "-m", "scripts.run_dependence_localization", "--config", config,
            "--checkpoint", str(checkpoint), "--calibration", str(calibration), "--dataset", "cub",
            "--root", DATASETS["cub"][0], "--samples", "100", "--randomize-backbone"])
        records.append({"architecture": architecture, "dataset": "cub", "randomized": True, "source_run": source_run, "run_id": run_id}); save(records, output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
