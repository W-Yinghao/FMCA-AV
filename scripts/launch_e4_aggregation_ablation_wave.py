#!/usr/bin/env python3
"""Submit the preregistered E4 parent/head alignment screening matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG = "configs/ssl/cifar10_smoke.json"


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def submit(name: str, override: dict[str, object], seed: int) -> str:
    command = [
        "python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", "--",
        "env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":")),
        f"FMCA_SEED_OVERRIDE={seed}", "bash", "scripts/run_fmca_pipeline.sh", "--config", CONFIG,
    ]
    while True:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def main() -> int:
    run_dir = Path(os.environ["FMCA_HARNESS_RUN_DIR"])
    artifacts = run_dir / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True)
    time.sleep(POLL_SECONDS)
    records = []
    for views in (2, 8):
        for aggregation in ("first", "mean", "deepsets", "concat"):
            for seed_index, seed in enumerate((20268001, 20268002, 20268003), 1):
                model: dict[str, object] = {"parent_feature_source": "backbone", "parent_aggregation": aggregation}
                if aggregation == "concat":
                    model["f_head_hidden_dims"] = [512]
                override = {"data": {"num_views": views}, "model": model}
                name = f"e4-cifar10-{aggregation}-m{views}-seed{seed_index}"
                records.append({"axis": "aggregation", "views": views, "setting": aggregation, "seed": seed, "run_id": submit(name, override, seed)})
                temporary = artifacts / "submitted.json.tmp"; temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(artifacts / "submitted.json")
    variants = {
        "separate_heads": {"parent_feature_source": "backbone", "parent_aggregation": "mean", "shared_head": False},
        "shared_head": {"parent_feature_source": "backbone", "parent_aggregation": "mean", "shared_head": True},
        "stop_f": {"parent_feature_source": "backbone", "parent_aggregation": "mean", "stop_gradient": "f"},
        "stop_g": {"parent_feature_source": "backbone", "parent_aggregation": "mean", "stop_gradient": "g"},
        "parent_from_g": {"parent_feature_source": "g", "parent_aggregation": "mean"},
    }
    for index, (setting, model) in enumerate(variants.items()):
        seed = 20268100 + index
        override = {"data": {"num_views": 8}, "model": model}
        name = f"e4-cifar10-{setting}-m8"
        records.append({"axis": "head_or_source", "views": 8, "setting": setting, "seed": seed, "run_id": submit(name, override, seed)})
        temporary = artifacts / "submitted.json.tmp"; temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(artifacts / "submitted.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
