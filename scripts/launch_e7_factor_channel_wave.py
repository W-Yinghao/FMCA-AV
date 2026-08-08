#!/usr/bin/env python3
"""Submit factor-dataset channel-swap training and spectral probes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
BATCH_LIMIT = 2
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
DATASETS = {
    "dsprites": "configs/ssl/dsprites_smoke.json", "shapes3d": "configs/ssl/shapes3d_smoke.json",
    "smallnorb": "configs/ssl/smallnorb_smoke.json", "mpi3d_toy": "configs/ssl/mpi3d_toy_smoke.json",
    "mpi3d_realistic": "configs/ssl/mpi3d_realistic_smoke.json", "mpi3d_real": "configs/ssl/mpi3d_real_smoke.json",
}
BASE = {
    "min_scale": 1.0, "max_scale": 1.0, "color_jitter_probability": 0.0,
    "grayscale_probability": 0.0, "gaussian_blur_probability": 0.0,
    "horizontal_flip_probability": 0.0, "rotation_degrees": 0.0, "additive_noise_std": 0.0,
}
CHANNELS = {
    "crop": {"min_scale": 0.2},
    "color": {"color_jitter_probability": 1.0, "color_jitter_strength": 1.0},
    "rotation": {"rotation_degrees": 90.0},
    "blur": {"gaussian_blur_probability": 1.0, "gaussian_blur_sigma": 2.0},
    "grayscale": {"grayscale_probability": 1.0},
    "noise": {"additive_noise_std": 0.2},
}


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def state(run_id: str) -> str:
    return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def wait_all(run_ids: list[str]) -> None:
    while True:
        refresh(); values = {run_id: state(run_id) for run_id in run_ids}
        failed = {run_id: value for run_id, value in values.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failed: raise RuntimeError("factor channel job failed: " + json.dumps(failed, sort_keys=True))
        if all(value == "SUCCEEDED" for value in values.values()): return
        time.sleep(POLL_SECONDS)


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
    time.sleep(POLL_SECONDS)
    submitted_path = artifacts / "submitted.json"
    records = json.loads(submitted_path.read_text(encoding="utf-8")) if submitted_path.is_file() else []
    existing = {(str(record["dataset"]), str(record["channel"])) for record in records}
    train_batch: list[str] = []
    for dataset_index, (dataset, config) in enumerate(DATASETS.items()):
        for channel_index, (channel, channel_values) in enumerate(CHANNELS.items()):
            if (dataset, channel) in existing:
                continue
            augmentation = {**BASE, **channel_values}
            override = {"experiment": {"name": f"e7-{dataset}-{channel}"}, "data": {"augmentation": augmentation}}
            seed = 20272000 + dataset_index * 100 + channel_index
            name = f"e7-{dataset}-{channel}-32step"
            argv = ["python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", "--", "env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":")), f"FMCA_SEED_OVERRIDE={seed}", "bash", "scripts/run_fmca_pipeline.sh", "--config", config]
            run_id = submit(argv); records.append({"dataset": dataset, "channel": channel, "config": config, "seed": seed, "override": override, "train_run": run_id})
            existing.add((dataset, channel))
            save(records, submitted_path)
            train_batch.append(run_id)
            if len(train_batch) >= BATCH_LIMIT:
                wait_all(train_batch)
                train_batch.clear()
    if train_batch:
        wait_all(train_batch)
    time.sleep(POLL_SECONDS)
    repair_failed(records, "train_run", submitted_path)
    wait_all([str(record["train_run"]) for record in records])
    probe_batch: list[str] = []
    for record in records:
        if record.get("probe_run"):
            continue
        train_run = str(record["train_run"]); result = json.loads((Path("runs") / train_run / "artifacts" / "train_result.json").read_text(encoding="utf-8"))
        checkpoint = result.get("best_checkpoint") or result.get("last_checkpoint")
        command = [
            "python3", "-m", "harness.cli", "submit", "--name", f"e7-{record['dataset']}-{record['channel']}-factor-probe", "--gpus", "1", "--",
            PYTHON, "-m", "scripts.run_factor_spectral_probe", "--config", str(record["config"]),
            "--checkpoint", str(checkpoint), "--calibration", f"runs/{train_run}/artifacts/calibration.pt",
            "--train-samples", "5000",
            "--test-samples", "2000", "--random-repeats", "5", "--rotation-repeats", "2", "--device", "cuda",
        ]
        probe_run = submit(command); record["probe_run"] = probe_run
        save(records, submitted_path)
        probe_batch.append(probe_run)
        if len(probe_batch) >= BATCH_LIMIT:
            wait_all(probe_batch)
            probe_batch.clear()
    if probe_batch:
        wait_all(probe_batch)
    time.sleep(POLL_SECONDS)
    repair_failed(records, "probe_run", submitted_path)
    wait_all([str(record["probe_run"]) for record in records])
    return 0


if __name__ == "__main__": raise SystemExit(main())
