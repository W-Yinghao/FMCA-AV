#!/usr/bin/env python3
"""Launch two additional seeds for factor-spectrum stability analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
PREREQUISITE = "20260807-061336_continue-factor-wave-fixed"
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIGS = {
    "dsprites": "configs/ssl/dsprites_smoke.json", "shapes3d": "configs/ssl/shapes3d_smoke.json",
    "smallnorb": "configs/ssl/smallnorb_smoke.json", "mpi3d_toy": "configs/ssl/mpi3d_toy_smoke.json",
}


def refresh() -> None: subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)
def state(run_id: str) -> str: return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def wait_all(run_ids: list[str]) -> None:
    while True:
        refresh(); values = {run_id: state(run_id) for run_id in run_ids}
        failed = {run_id: value for run_id, value in values.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failed: raise RuntimeError("factor stability job failed: " + json.dumps(failed, sort_keys=True))
        if all(value == "SUCCEEDED" for value in values.values()): return
        time.sleep(POLL_SECONDS)


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def save(records: list[dict[str, object]], output: Path) -> None:
    temporary = output.with_suffix(".tmp"); temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)


def main() -> int:
    artifacts = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True); output = artifacts / "submitted.json"
    time.sleep(POLL_SECONDS); wait_all([PREREQUISITE]); records = []
    for dataset_index, (dataset, config) in enumerate(CONFIGS.items()):
        for seed_index in (2, 3):
            seed = 20277000 + dataset_index * 10 + seed_index; name = f"e7-{dataset}-stability-seed{seed_index}"
            override = {"experiment": {"name": name}}
            train_run = submit(["python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1", "--", "env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override, separators=(",", ":")), f"FMCA_SEED_OVERRIDE={seed}", "bash", "scripts/run_fmca_pipeline.sh", "--config", config])
            records.append({"dataset": dataset, "config": config, "seed_index": seed_index, "seed": seed, "train_run": train_run}); save(records, output)
    time.sleep(POLL_SECONDS); wait_all([str(record["train_run"]) for record in records])
    for record in records:
        train_run = str(record["train_run"]); result = json.loads((Path("runs") / train_run / "artifacts" / "train_result.json").read_text(encoding="utf-8")); checkpoint = result.get("best_checkpoint") or result.get("last_checkpoint")
        if not checkpoint: raise RuntimeError(f"checkpoint missing for {train_run}")
        probe_run = submit([
            "python3", "-m", "harness.cli", "submit", "--name", f"e7-{record['dataset']}-stability-seed{record['seed_index']}-factor-probe", "--gpus", "1", "--",
            PYTHON, "-m", "scripts.run_factor_spectral_probe", "--config", str(record["config"]), "--checkpoint", str(checkpoint),
            "--calibration", f"runs/{train_run}/artifacts/calibration.pt", "--train-samples", "5000", "--test-samples", "2000", "--random-repeats", "5", "--rotation-repeats", "2", "--device", "cuda",
        ])
        record["probe_run"] = probe_run; save(records, output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
