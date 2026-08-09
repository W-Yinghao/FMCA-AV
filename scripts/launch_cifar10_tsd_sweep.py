#!/usr/bin/env python3
"""Submit and monitor the preregistered CIFAR-10 augmentation-severity matrix."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
BATCH_LIMIT = 2
BASELINE_WATCHER = "20260807-060401_continue-cifar10-baseline-screening-fixed"
VALIDATION_RUN = "20260807-055116_validate-cifar-augmentation-severity"
CONFIG = "configs/ssl/cifar10_smoke.json"
SEEDS = [20261101, 20261102, 20261103, 20261104, 20261105]
LEVELS = {
    "crop": [
        {"min_scale": value} for value in (1.0, 0.8, 0.6, 0.4, 0.25, 0.15, 0.08)
    ],
    "color": [
        {"color_jitter_probability": 1.0, "color_jitter_strength": value}
        for value in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
    ],
    "blur": [
        {"gaussian_blur_probability": 0.0 if value == 0 else 1.0, "gaussian_blur_sigma": value}
        for value in (0.0, 0.2, 0.5, 0.8, 1.2, 1.6, 2.0)
    ],
    "rotation": [
        {"rotation_degrees": value} for value in (0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 90.0)
    ],
    "grayscale": [
        {"grayscale_probability": value} for value in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    ],
    "noise": [
        {"additive_noise_std": value} for value in (0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3)
    ],
}


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def state(run_id: str) -> str:
    with (Path("runs") / run_id / "status.json").open(encoding="utf-8") as handle:
        return str(json.load(handle)["state"])


def wait_success(run_ids: list[str]) -> None:
    while True:
        refresh()
        values = {run_id: state(run_id) for run_id in run_ids}
        failed = {run_id: value for run_id, value in values.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failed:
            raise RuntimeError("prerequisite or sweep job failed: " + json.dumps(failed, sort_keys=True))
        if all(value == "SUCCEEDED" for value in values.values()):
            return
        time.sleep(POLL_SECONDS)


def submit(name: str, overrides: dict[str, object], seed: int) -> str:
    environment_override = json.dumps(
        {
            "experiment": {"name": name},
            "data": {"augmentation": overrides},
        },
        separators=(",", ":"),
    )
    command = [
        "python3", "-m", "harness.cli", "submit",
        "--name", name, "--gpus", "1", "--",
        "env", f"FMCA_CONFIG_OVERRIDES={environment_override}", f"FMCA_SEED_OVERRIDE={seed}",
        "bash", "scripts/run_fmca_pipeline.sh", "--config", CONFIG,
    ]
    while True:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode == 0:
            return completed.stdout.strip()
        if "GPU limit exceeded" not in completed.stderr:
            raise RuntimeError(completed.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def save(records: list[dict[str, object]], output: Path) -> None:
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def retry_with_capacity(run_id: str) -> str:
    while True:
        completed = subprocess.run(
            ["python3", "-m", "harness.cli", "retry", "--run", run_id],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
        if "GPU limit exceeded" not in completed.stderr:
            raise RuntimeError(completed.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def main() -> int:
    run_dir = Path(__import__("os").environ["FMCA_HARNESS_RUN_DIR"])
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    time.sleep(POLL_SECONDS)
    wait_success([BASELINE_WATCHER, VALIDATION_RUN])
    output = artifacts / "submitted.json"
    records = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else []
    for record in records:
        run_id = str(record["run_id"])
        if state(run_id) in {"FAILED", "STOPPED", "BLOCKED"}:
            record["run_id"] = retry_with_capacity(run_id)
            save(records, output)
    if records:
        wait_success([str(record["run_id"]) for record in records])
    existing = {(str(record["channel"]), int(record["level"]), int(record["seed"])) for record in records}
    inflight: list[str] = []
    for channel, configurations in LEVELS.items():
        for level, overrides in enumerate(configurations):
            for seed_index, seed in enumerate(SEEDS, start=1):
                if (channel, level, seed) in existing:
                    continue
                name = f"tsd-cifar10-{channel}-level{level}-seed{seed_index}"
                run_id = submit(name, overrides, seed)
                records.append({"channel": channel, "level": level, "seed": seed, "run_id": run_id, "overrides": overrides})
                existing.add((channel, level, seed))
                save(records, output)
                inflight.append(run_id)
                if len(inflight) >= BATCH_LIMIT:
                    wait_success(inflight)
                    inflight.clear()
    if inflight:
        wait_success(inflight)
    time.sleep(POLL_SECONDS)
    wait_success([record["run_id"] for record in records])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
