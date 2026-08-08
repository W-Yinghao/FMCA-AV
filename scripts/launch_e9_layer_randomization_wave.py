#!/usr/bin/env python3
"""Run progressive layer-randomization sanity checks on dependence maps."""

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
    "vgg16_bn": ("20260807-092551_imagenet100-vgg16-bn-32step-smoke", "configs/ssl/imagenet100_smoke.json", {"model": {"backbone": "vgg16_bn"}, "data": {"batch_size": 32}}),
}
CUB_ROOT = "/projects/EEG-foundation-model/yinghao/FMCA-AV/cub"


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_state(run_id: str) -> str:
    return str(read(Path("runs") / run_id / "status.json")["state"])


def wait_all(run_ids: list[str]) -> None:
    while True:
        time.sleep(POLL_SECONDS); refresh(); states = {run_id: run_state(run_id) for run_id in run_ids}
        failures = {key: value for key, value in states.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failures: raise RuntimeError("layer-randomization source/job failed: " + json.dumps(failures, sort_keys=True))
        if all(value == "SUCCEEDED" for value in states.values()): return


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def main() -> int:
    artifacts = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True)
    output = artifacts / "e9_layer_submitted.json"; time.sleep(POLL_SECONDS); wait_all([source[0] for source in SOURCES.values()])
    records = []
    for architecture, (source_run, config, override) in SOURCES.items():
        train_result = read(Path("runs") / source_run / "artifacts" / "train_result.json")
        checkpoint = train_result.get("best_checkpoint") or train_result.get("last_checkpoint")
        calibration = Path("runs") / source_run / "artifacts" / "calibration.pt"
        environment = ["env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":"))] if override else []
        composition_run = submit([
            "python3", "-m", "harness.cli", "submit", "--name", f"e9-{architecture}-cnn-direct-recursive-composition",
            "--gpus", "1", "--profile", "imagenet", "--", *environment, PYTHON, "-m", "scripts.run_cnn_composition_maps",
            "--config", config, "--checkpoint", str(checkpoint), "--model-type", "fmca", "--root", CUB_ROOT,
            "--calibration-samples", "50", "--evaluation-samples", "50",
        ])
        records.append({"architecture": architecture, "stage": "composition", "source_run": source_run, "run_id": composition_run})
        temporary = output.with_suffix(".tmp"); temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
        for stage in range(1, 5):
            name = f"e9-sanity-{architecture}-randomize-stage{stage}-cub"
            run_id = submit([
                "python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", "--profile", "imagenet", "--",
                *environment, PYTHON, "-m", "scripts.run_dependence_localization", "--config", config,
                "--checkpoint", str(checkpoint), "--calibration", str(calibration), "--dataset", "cub",
                "--root", CUB_ROOT, "--samples", "100", "--randomize-from-stage", str(stage),
            ])
            records.append({"architecture": architecture, "stage": stage, "source_run": source_run, "run_id": run_id})
            temporary = output.with_suffix(".tmp"); temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    supervised_run = "20260807-074105_supervised-imagenet1k-reference-32step"
    supervised_result = read(Path("runs") / supervised_run / "artifacts" / "supervised_result.json")
    supervised_checkpoint = supervised_result.get("best_checkpoint") or supervised_result.get("last_checkpoint")
    supervised_composition = submit([
        "python3", "-m", "harness.cli", "submit", "--name", "e9-supervised-resnet50-cnn-direct-recursive-composition",
        "--gpus", "1", "--profile", "imagenet", "--", PYTHON, "-m", "scripts.run_cnn_composition_maps",
        "--config", "configs/ssl/imagenet1k_smoke.json", "--checkpoint", str(supervised_checkpoint),
        "--model-type", "supervised", "--root", CUB_ROOT, "--calibration-samples", "50", "--evaluation-samples", "50",
    ])
    records.append({"architecture": "resnet50", "stage": "composition", "method": "supervised", "source_run": supervised_run, "run_id": supervised_composition})
    temporary = output.with_suffix(".tmp"); temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    wait_all([str(record["run_id"]) for record in records])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
