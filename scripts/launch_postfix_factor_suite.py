#!/usr/bin/env python3
"""Restartable post-fix E7 factor training and spectral-probe suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


POLL_SECONDS = 300
BATCH_LIMIT = 2
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"}
CONFIGS = {
    "dsprites": "configs/ssl/dsprites_smoke.json",
    "shapes3d": "configs/ssl/shapes3d_smoke.json",
    "smallnorb": "configs/ssl/smallnorb_smoke.json",
    "mpi3d_toy": "configs/ssl/mpi3d_toy_smoke.json",
    "mpi3d_realistic": "configs/ssl/mpi3d_realistic_smoke.json",
    "mpi3d_real": "configs/ssl/mpi3d_real_smoke.json",
}
BASE_AUGMENTATION = {
    "min_scale": 1.0, "max_scale": 1.0, "color_jitter_probability": 0.0,
    "grayscale_probability": 0.0, "gaussian_blur_probability": 0.0,
    "horizontal_flip_probability": 0.0, "rotation_degrees": 0.0,
    "additive_noise_std": 0.0,
}
CHANNELS = {
    "crop": {"min_scale": 0.2},
    "color": {"color_jitter_probability": 1.0, "color_jitter_strength": 1.0},
    "rotation": {"rotation_degrees": 90.0},
    "blur": {"gaussian_blur_probability": 1.0, "gaussian_blur_sigma": 2.0},
    "grayscale": {"grayscale_probability": 1.0},
    "noise": {"additive_noise_std": 0.2},
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def run_state(run_id: str) -> str:
    return str(read(Path("runs") / run_id / "status.json")["state"])


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def wait_all(run_ids: list[str], label: str) -> None:
    if not run_ids:
        return
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        states = {run_id: run_state(run_id) for run_id in run_ids}
        failures = {run_id: state for run_id, state in states.items() if state in TERMINAL and state != "SUCCEEDED"}
        if failures:
            raise RuntimeError(f"{label} failed: " + json.dumps(failures, sort_keys=True))
        if all(state == "SUCCEEDED" for state in states.values()):
            return


def plan() -> list[dict]:
    records = []
    for dataset_index, (dataset, config) in enumerate(CONFIGS.items()):
        for seed_index in (1, 2, 3):
            seed = 20310000 + dataset_index * 100 + seed_index
            records.append({
                "key": f"{dataset}:default:seed{seed_index}", "dataset": dataset,
                "channel": "default", "seed_index": seed_index, "seed": seed,
                "config": config, "augmentation": BASE_AUGMENTATION,
                "probe_budget": "full", "train_run": "", "probe_run": "",
            })
        for channel_index, (channel, values) in enumerate(CHANNELS.items(), 1):
            seed = 20310000 + dataset_index * 100 + 50 + channel_index
            records.append({
                "key": f"{dataset}:{channel}:seed1", "dataset": dataset,
                "channel": channel, "seed_index": 1, "seed": seed, "config": config,
                "augmentation": {**BASE_AUGMENTATION, **values},
                "probe_budget": "channel", "train_run": "", "probe_run": "",
            })
    return records


def load_state(path: Path) -> dict:
    if path.is_file():
        state = read(path)
        if state.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            raise RuntimeError(f"refusing legacy factor state: {path}")
        return state
    return {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "state": "RUNNING", "chain_runs": [], "records": plan(), "summary_run": "",
    }


def validate_training(record: dict) -> tuple[str, str]:
    run_id = str(record["train_run"])
    payload = read(Path("runs") / run_id / "artifacts" / "train_result.json")
    if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing legacy factor checkpoint from {run_id}")
    checkpoint = str(payload.get("best_checkpoint") or payload.get("last_checkpoint") or "")
    calibration = Path("runs") / run_id / "artifacts" / "calibration.pt"
    if not checkpoint or not Path(checkpoint).is_file() or not calibration.is_file():
        raise RuntimeError(f"factor source {run_id} lacks checkpoint or calibration")
    return checkpoint, str(calibration)


def submit_training(record: dict) -> str:
    override = {
        "experiment": {"name": f"postfix-e7-{record['dataset']}-{record['channel']}-seed{record['seed_index']}"},
        "data": {"augmentation": record["augmentation"]},
    }
    return submit([
        "python3", "-m", "harness.cli", "submit", "--name",
        f"postfix-e7-{record['dataset']}-{record['channel']}-seed{record['seed_index']}-train",
        "--gpus", "1", "--", "env",
        "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":")),
        f"FMCA_SEED_OVERRIDE={record['seed']}",
        "bash", "scripts/run_fmca_pipeline.sh", "--config", str(record["config"]),
    ])


def submit_probe(record: dict) -> str:
    checkpoint, calibration = validate_training(record)
    if record["probe_budget"] == "full":
        samples = ("20000", "5000", "100", "20")
    else:
        samples = ("5000", "2000", "5", "2")
    return submit([
        "python3", "-m", "harness.cli", "submit", "--name",
        f"postfix-e7-{record['dataset']}-{record['channel']}-seed{record['seed_index']}-probe",
        "--gpus", "1", "--", PYTHON, "-m", "scripts.run_factor_spectral_probe",
        "--config", str(record["config"]), "--checkpoint", checkpoint,
        "--calibration", calibration, "--train-samples", samples[0],
        "--test-samples", samples[1], "--random-repeats", samples[2],
        "--rotation-repeats", samples[3], "--device", "cuda",
    ])


def run_phase(state: dict, state_path: Path, field: str, operation, label: str) -> None:
    records = list(state["records"])
    for start in range(0, len(records), BATCH_LIMIT):
        batch = records[start:start + BATCH_LIMIT]
        run_ids = []
        for record in batch:
            run_id = str(record.get(field, ""))
            if run_id and run_state(run_id) == "SUCCEEDED":
                run_ids.append(run_id)
                continue
            if run_id and run_state(run_id) not in TERMINAL:
                run_ids.append(run_id)
                continue
            record[field] = operation(record)
            run_ids.append(str(record[field]))
            state["records"] = records
            save(state_path, state)
        wait_all(run_ids, label)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-file",
        default=f"results/orchestration/factor_suite_{SCIENTIFIC_CORRECTNESS_VERSION}.json",
    )
    args = parser.parse_args()
    state_path = Path(args.state_file)
    state = load_state(state_path)
    chain_runs = list(state.get("chain_runs", []))
    current_run = os.environ["FMCA_HARNESS_RUN_ID"]
    if current_run not in chain_runs:
        chain_runs.append(current_run)
    state["chain_runs"] = chain_runs
    save(state_path, state)

    run_phase(state, state_path, "train_run", submit_training, "factor training")
    run_phase(state, state_path, "probe_run", submit_probe, "factor probing")

    summary_run = str(state.get("summary_run", ""))
    if summary_run and run_state(summary_run) not in TERMINAL:
        wait_all([summary_run], "factor summary")
    if not summary_run or run_state(summary_run) != "SUCCEEDED":
        summary_run = submit([
            "python3", "-m", "harness.cli", "submit", "--name", "postfix-e7-factor-summary",
            "--gpus", "0", "--", PYTHON, "-m", "scripts.summarize_factor_probes",
        ])
        state["summary_run"] = summary_run
        save(state_path, state)
    wait_all([summary_run], "factor summary")
    state["state"] = "SUCCEEDED"
    save(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
