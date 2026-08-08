#!/usr/bin/env python3
"""Recover waves interrupted while the harness CLI was temporarily invalid."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from scripts import launch_cifar10_tsd_sweep as tsd


POLL_SECONDS = 300
OLD_E3 = Path("runs/20260807-062417_launch-e3-cifar-numerics-wave/artifacts/submitted.json")
CONFIG = "configs/ssl/cifar10_smoke.json"
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
COCO_CHECKPOINT = "runs/20260807-052202_imagenet1k-lightning-32step-smoke/artifacts/checkpoints/best-000-3.492188.ckpt"
ORIGINAL_MASK_RUN = "20260807-073728_coco2017-fmca-maskrcnn-16step-smoke-fixed-v2"


def refresh() -> None: subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def designs() -> list[tuple[str, dict[str, object]]]:
    values: list[tuple[str, dict[str, object]]] = []
    for objective in ("trace", "logdet"): values.append((f"objective-{objective}", {"objective": {"name": objective}}))
    for ridge in (1e-2, 1e-3, 1e-4, 1e-5): values.append((f"ridge-{ridge:g}", {"objective": {"ridge": ridge}}))
    for batch in (64, 128, 256, 512): values.append((f"batch-{batch}", {"data": {"batch_size": batch}}))
    for dimension in (32, 64, 128, 256): values.append((f"k-{dimension}", {"model": {"feature_dim": dimension}}))
    for precision in ("32-true", "16-mixed"): values.append((f"precision-{precision}", {"trainer": {"precision": precision}}))
    for tag, hidden in (("small", [256]), ("base", [512]), ("large", [1024, 1024])): values.append((f"capacity-{tag}", {"model": {"head_hidden_dims": hidden}}))
    ridges = (1e-2, 1e-3, 1e-4, 1e-5); dimensions = (32, 64, 128, 256); batches = (64, 128, 256, 512)
    for cell in range(16):
        values.append((f"fractional-{cell:02d}", {"objective": {"name": ("trace", "logdet")[cell % 2], "ridge": ridges[cell % 4]}, "model": {"feature_dim": dimensions[(cell * 3) % 4]}, "data": {"batch_size": batches[(cell // 4 + cell) % 4]}}))
    return values


def atomic_records(path: Path, records: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(".tmp"); temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(path)


def main() -> int:
    artifacts = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True)
    mask_run = ORIGINAL_MASK_RUN
    if json.loads((Path("runs") / mask_run / "status.json").read_text(encoding="utf-8"))["state"] != "SUCCEEDED":
        mask_run = submit([
            "python3", "-m", "harness.cli", "submit", "--name", "coco2017-fmca-maskrcnn-16step-smoke-fixed-v2", "--gpus", "1", "--profile", "imagenet", "--",
            PYTHON, "-m", "scripts.run_coco_transfer", "--config", "configs/ssl/imagenet1k_smoke.json", "--checkpoint", COCO_CHECKPOINT,
            "--root", "/projects/EEG-foundation-model/yinghao/FMCA-AV/coco", "--task", "instance_segmentation", "--max-steps", "16", "--train-images", "1000", "--val-images", "100",
        ])
    (artifacts / "coco_mask_run.txt").write_text(mask_run + "\n", encoding="utf-8")
    recovered_path = artifacts / "e3_submitted.json"
    existing = json.loads(recovered_path.read_text(encoding="utf-8")) if recovered_path.is_file() else json.loads(OLD_E3.read_text(encoding="utf-8"))
    seen = {str(record["tag"]) for record in existing}; combined = list(existing)
    for index, (tag, override) in enumerate(designs()):
        if tag in seen: continue
        seed = 20269000 + index
        run_id = submit([
            "python3", "-m", "harness.cli", "submit", "--name", f"e3-cifar10-{tag}-recovered", "--gpus", "1", "--",
            "env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":")), f"FMCA_SEED_OVERRIDE={seed}",
            "bash", "scripts/run_fmca_pipeline.sh", "--config", CONFIG,
        ])
        combined.append({"tag": tag, "seed": seed, "override": override, "run_id": run_id, "recovered": True}); atomic_records(recovered_path, combined)
    # This function uses the same harness run directory, writes submitted.json,
    # polls every 300 s, and waits for all 210 children before returning.
    return int(tsd.main())


if __name__ == "__main__": raise SystemExit(main())
