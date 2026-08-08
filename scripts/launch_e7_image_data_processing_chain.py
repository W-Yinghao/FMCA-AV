#!/usr/bin/env python3
"""Run a preregistered cumulative crop-color-blur data-processing chain."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
RECOVERY = "20260807-073411_recover-interrupted-e3-tsd-coco-v2"
CONFIG = "configs/ssl/cifar10_smoke.json"
STAGES = (
    {"min_scale": 1.0, "color_jitter_probability": 0.0, "gaussian_blur_probability": 0.0},
    {"min_scale": 0.8, "color_jitter_probability": 0.0, "gaussian_blur_probability": 0.0},
    {"min_scale": 0.6, "color_jitter_probability": 1.0, "color_jitter_strength": 0.1, "gaussian_blur_probability": 0.0},
    {"min_scale": 0.4, "color_jitter_probability": 1.0, "color_jitter_strength": 0.3, "gaussian_blur_probability": 1.0, "gaussian_blur_sigma": 0.5},
    {"min_scale": 0.25, "color_jitter_probability": 1.0, "color_jitter_strength": 0.5, "gaussian_blur_probability": 1.0, "gaussian_blur_sigma": 1.2},
    {"min_scale": 0.15, "color_jitter_probability": 1.0, "color_jitter_strength": 0.7, "gaussian_blur_probability": 1.0, "gaussian_blur_sigma": 1.6},
    {"min_scale": 0.08, "color_jitter_probability": 1.0, "color_jitter_strength": 1.0, "gaussian_blur_probability": 1.0, "gaussian_blur_sigma": 2.0},
)


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def state(run_id: str) -> str:
    return str(read(Path("runs") / run_id / "status.json")["state"])


def checkpoint(run_id: str) -> str:
    value = read(Path("runs") / run_id / "artifacts" / "train_result.json")
    candidate = value.get("best_checkpoint") or value.get("last_checkpoint")
    if not candidate or not Path(str(candidate)).is_file():
        raise RuntimeError(f"processing-chain checkpoint missing: {run_id}")
    return str(candidate)


def wait_all(run_ids: list[str]) -> None:
    while True:
        time.sleep(POLL_SECONDS); refresh(); states = {run_id: state(run_id) for run_id in run_ids}
        failures = {key: value for key, value in states.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failures: raise RuntimeError("image data-processing chain failed: " + json.dumps(failures, sort_keys=True))
        if all(value == "SUCCEEDED" for value in states.values()): return


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def main() -> int:
    artifacts = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True)
    output = artifacts / "image_chain_submitted.json"; time.sleep(POLL_SECONDS); wait_all([RECOVERY]); records = []
    for stage, augmentation in enumerate(STAGES):
        for seed_index in range(1, 6):
            seed = 20279000 + seed_index
            name = f"tsd-cifar10-processing-chain-stage{stage}-seed{seed_index}"
            override = {"experiment": {"name": name}, "data": {"augmentation": augmentation}}
            run_id = submit([
                "python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", "--",
                "env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":")),
                f"FMCA_SEED_OVERRIDE={seed}", "bash", "scripts/run_fmca_pipeline.sh", "--config", CONFIG,
            ])
            records.append({"stage": stage, "seed_index": seed_index, "seed": seed, "augmentation": augmentation, "run_id": run_id})
            temporary = output.with_suffix(".tmp"); temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    wait_all([str(record["run_id"]) for record in records])
    for record in records:
        probe_run = submit([
            "python3", "-m", "harness.cli", "submit", "--name",
            f"{record['run_id']}-processing-chain-utility-linear-probe", "--gpus", "1", "--",
            "env", f"FMCA_SEED_OVERRIDE={record['seed']}", PYTHON, "-m", "fmca_av.cli", "linear-probe",
            "--config", CONFIG, "--checkpoint", checkpoint(str(record["run_id"])),
        ])
        record["probe_run"] = probe_run
        temporary = output.with_suffix(".tmp"); temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    wait_all([str(record["probe_run"]) for record in records])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
