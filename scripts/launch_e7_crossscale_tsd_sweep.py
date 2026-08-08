#!/usr/bin/env python3
"""Launch CIFAR-100 full-severity and ImageNet100 three-level TSD sweeps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
BATCH_LIMIT = 2
PREREQUISITE = "20260807-060445_launch-cifar10-tsd-full-severity-sweep-fixed"
LEVELS = {
    "crop": [{"min_scale": value} for value in (1.0, 0.8, 0.6, 0.4, 0.25, 0.15, 0.08)],
    "color": [{"color_jitter_probability": 1.0, "color_jitter_strength": value} for value in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)],
    "blur": [{"gaussian_blur_probability": 0.0 if value == 0 else 1.0, "gaussian_blur_sigma": value} for value in (0.0, 0.2, 0.5, 0.8, 1.2, 1.6, 2.0)],
    "rotation": [{"rotation_degrees": value} for value in (0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 90.0)],
    "grayscale": [{"grayscale_probability": value} for value in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)],
    "noise": [{"additive_noise_std": value} for value in (0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3)],
}


def refresh() -> None: subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)
def state(run_id: str) -> str: return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def wait_success(run_ids: str | list[str]) -> None:
    pending = [run_ids] if isinstance(run_ids, str) else list(run_ids)
    while True:
        refresh()
        states = {run_id: state(run_id) for run_id in pending}
        failures = {run_id: value for run_id, value in states.items()
                    if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failures:
            detail = ", ".join(f"{run_id}={value}" for run_id, value in failures.items())
            raise RuntimeError(f"cross-scale child failed: {detail}")
        if all(value == "SUCCEEDED" for value in states.values()):
            return
        time.sleep(POLL_SECONDS)


def submit(name: str, config: str, profile: list[str], overrides: dict[str, object], seed: int) -> str:
    environment = json.dumps({"experiment": {"name": name}, "data": {"augmentation": overrides}}, separators=(",", ":"))
    command = ["python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", *profile, "--", "env", "FMCA_CONFIG_OVERRIDES=" + environment, f"FMCA_SEED_OVERRIDE={seed}", "bash", "scripts/run_fmca_pipeline.sh", "--config", config]
    while True:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def save(records: list[dict[str, object]], output: Path) -> None:
    temporary = output.with_suffix(".tmp"); temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)


def retry_with_capacity(run_id: str) -> str:
    command = ["python3", "-m", "harness.cli", "retry", "--run", run_id]
    while True:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def main(phase: str | None = None) -> int:
    if phase is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--phase",
            choices=("all", "non-imagenet", "imagenet"),
            default="all",
            help="Select a scheduling phase so ImageNet work can be deferred.",
        )
        phase = parser.parse_args().phase
    artifacts = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True); output = artifacts / "submitted.json"
    time.sleep(POLL_SECONDS); wait_success(PREREQUISITE)
    records = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else []
    refresh()
    for record in records:
        run_id = str(record["run_id"])
        if state(run_id) in {"FAILED", "STOPPED", "BLOCKED"}:
            record["run_id"] = retry_with_capacity(run_id)
            save(records, output)
    existing = {(str(record["dataset"]), str(record["channel"]), int(record["level"]),
                 int(record["seed"])) for record in records}
    if records:
        wait_success([str(record["run_id"]) for record in records])
    inflight: list[str] = []
    if phase in {"all", "non-imagenet"}:
        for channel, levels in LEVELS.items():
            for level, overrides in enumerate(levels):
                for seed_index, seed in enumerate((20278101, 20278102, 20278103, 20278104, 20278105), 1):
                    if ("cifar100", channel, level, seed) in existing: continue
                    name = f"tsd-cifar100-{channel}-level{level}-seed{seed_index}"; run_id = submit(name, "configs/ssl/cifar100_smoke.json", [], overrides, seed)
                    records.append({"dataset": "cifar100", "channel": channel, "level": level, "seed": seed, "overrides": overrides, "run_id": run_id}); save(records, output)
                    existing.add(("cifar100", channel, level, seed))
                    inflight.append(run_id)
                    if len(inflight) >= BATCH_LIMIT:
                        wait_success(inflight); inflight.clear()
    if phase in {"all", "imagenet"}:
        for channel, levels in LEVELS.items():
            for level in (0, 3, 6):
                overrides = levels[level]
                for seed_index, seed in enumerate((20278201, 20278202, 20278203), 1):
                    if ("imagenet100", channel, level, seed) in existing: continue
                    name = f"tsd-imagenet100-{channel}-level{level}-seed{seed_index}"; run_id = submit(name, "configs/ssl/imagenet100_smoke.json", ["--profile", "imagenet"], overrides, seed)
                    records.append({"dataset": "imagenet100", "channel": channel, "level": level, "seed": seed, "overrides": overrides, "run_id": run_id}); save(records, output)
                    existing.add(("imagenet100", channel, level, seed))
                    inflight.append(run_id)
                    if len(inflight) >= BATCH_LIMIT:
                        wait_success(inflight); inflight.clear()
    if inflight:
        wait_success(inflight)
    return 0


if __name__ == "__main__": raise SystemExit(main())
