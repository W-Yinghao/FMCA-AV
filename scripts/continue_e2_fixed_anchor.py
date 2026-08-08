#!/usr/bin/env python3
"""Run and render the corrected fixed-anchor E2 CIFAR variance protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG = "configs/ssl/cifar10_paper_concat_smoke.json"
CHECKPOINT = "/home/infres/yinwang/FMCA-AV/runs/20260807-042119_cifar10-gmean-matched-head-20epoch/artifacts/checkpoints/best-019-63.508625.ckpt"
OVERRIDE = {"model": {"parent_aggregation": "mean", "f_head_hidden_dims": [2552]}, "data": {"num_views": 9}}


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def run_state(run_id: str) -> str:
    return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def main() -> int:
    time.sleep(POLL_SECONDS)
    argv = [
        "python3", "-m", "harness.cli", "submit", "--name", "e2-cifar10-fixed-anchor-gradient-variance-500",
        "--gpus", "1", "--", "env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(OVERRIDE, separators=(",", ":")),
        PYTHON, "-m", "scripts.run_e2_cifar_gradient_variance", "--config", CONFIG,
        "--checkpoint", CHECKPOINT, "--repetitions", "500", "--fixed-anchor-parent",
    ]
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            run_id = result.stdout.strip()
            break
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()
    artifacts = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "gradient_run.txt").write_text(run_id + "\n", encoding="utf-8")
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        value = run_state(run_id)
        if value == "SUCCEEDED":
            break
        if value in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError(f"fixed-anchor E2 run {run_id} ended in {value}")
    source = Path("runs") / run_id / "artifacts" / "cifar_gradient_variance.json"
    subprocess.run([PYTHON, "-m", "scripts.render_e2_variance_assets", "--input", str(source), "--output-dir", "results/e2"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
