#!/usr/bin/env python3
"""Submit the E3 CIFAR numerical/objective fractional-factorial matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
CONFIG = "configs/ssl/cifar10_smoke.json"


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def submit(name: str, override: dict[str, object], seed: int) -> str:
    argv = [
        "python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", "--",
        "env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":")),
        f"FMCA_SEED_OVERRIDE={seed}", "bash", "scripts/run_fmca_pipeline.sh", "--config", CONFIG,
    ]
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def main() -> int:
    run_dir = Path(os.environ["FMCA_HARNESS_RUN_DIR"]); artifacts = run_dir / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True)
    time.sleep(POLL_SECONDS)
    designs: list[tuple[str, dict[str, object]]] = []
    for objective in ("trace", "logdet"):
        designs.append((f"objective-{objective}", {"objective": {"name": objective}}))
    for ridge in (1e-2, 1e-3, 1e-4, 1e-5):
        designs.append((f"ridge-{ridge:g}", {"objective": {"ridge": ridge}}))
    for batch in (64, 128, 256, 512):
        designs.append((f"batch-{batch}", {"data": {"batch_size": batch}}))
    for dimension in (32, 64, 128, 256):
        designs.append((f"k-{dimension}", {"model": {"feature_dim": dimension}}))
    for precision in ("32-true", "16-mixed"):
        designs.append((f"precision-{precision}", {"trainer": {"precision": precision}}))
    for tag, hidden in (("small", [256]), ("base", [512]), ("large", [1024, 1024])):
        designs.append((f"capacity-{tag}", {"model": {"head_hidden_dims": hidden}}))
    # A deterministic 16-cell balanced subset of objective x ridge x K x batch.
    ridges = (1e-2, 1e-3, 1e-4, 1e-5); dimensions = (32, 64, 128, 256); batches = (64, 128, 256, 512)
    for cell in range(16):
        objective = ("trace", "logdet")[cell % 2]
        ridge = ridges[cell % 4]
        dimension = dimensions[(cell * 3) % 4]
        batch = batches[(cell // 4 + cell) % 4]
        designs.append((f"fractional-{cell:02d}", {"objective": {"name": objective, "ridge": ridge}, "model": {"feature_dim": dimension}, "data": {"batch_size": batch}}))
    records = []
    for index, (tag, override) in enumerate(designs):
        seed = 20269000 + index
        run_id = submit(f"e3-cifar10-{tag}", override, seed)
        records.append({"tag": tag, "seed": seed, "override": override, "run_id": run_id})
        temporary = artifacts / "submitted.json.tmp"; temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(artifacts / "submitted.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
