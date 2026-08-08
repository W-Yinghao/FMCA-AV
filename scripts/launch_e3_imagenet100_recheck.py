#!/usr/bin/env python3
"""Short ImageNet-100 recheck of the best/reference and numerical stress cells."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG = "configs/ssl/imagenet100_smoke.json"
STATE_PATH = Path("results/orchestration/e3_imagenet100_recheck_state.json")
CELLS = {
    "reference": {
        "objective": {"name": "trace", "ridge": 1e-3},
        "model": {"feature_dim": 128},
        "trainer": {"precision": "bf16-mixed"},
    },
    "logdet": {
        "objective": {"name": "logdet", "ridge": 1e-3},
        "model": {"feature_dim": 128},
        "trainer": {"precision": "32-true"},
    },
    "stable-small": {
        "objective": {"name": "trace", "ridge": 1e-2},
        "model": {"feature_dim": 32},
        "trainer": {"precision": "32-true"},
    },
    "stress-large": {
        "objective": {"name": "trace", "ridge": 1e-5},
        "model": {"feature_dim": 256},
        "trainer": {"precision": "bf16-mixed"},
    },
}


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(state: dict[str, object]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def run_state(run_id: str) -> str:
    return str(read(Path("runs") / run_id / "status.json")["state"])


def submit(name: str, seed: int, override: dict[str, object]) -> str:
    payload = {
        **override,
        "experiment": {"name": name},
        "data": {"num_views": 8, "batch_size": 32},
        "trainer": {
            **dict(override.get("trainer", {})),
            "devices": 1, "strategy": "auto", "max_epochs": 2,
            "limit_train_batches": 64, "limit_val_batches": 8,
        },
        "optimizer": {"scheduler_t_max": 2},
    }
    argv = [
        "python3", "-m", "harness.cli", "submit", "--name", name,
        "--gpus", "1", "--profile", "imagenet", "--", PYTHON, "-m",
        "scripts.run_fmca_pipeline", "--config", CONFIG, "--seed", str(seed),
        "--overrides-json", json.dumps(payload, separators=(",", ":")), "--train-only",
    ]
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def wait_all(run_ids: list[str]) -> None:
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        states = {run_id: run_state(run_id) for run_id in run_ids}
        failures = {key: value for key, value in states.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failures:
            raise RuntimeError("ImageNet-100 E3 recheck failed: " + json.dumps(failures, sort_keys=True))
        if all(value == "SUCCEEDED" for value in states.values()):
            return


def main() -> int:
    state = read(STATE_PATH) if STATE_PATH.is_file() else {"state": "RUNNING", "chain_runs": [], "submitted": []}
    chain = list(state.get("chain_runs", []))
    current_chain_run = os.environ["FMCA_HARNESS_RUN_ID"]
    if current_chain_run not in chain: chain.append(current_chain_run)
    state["chain_runs"] = chain
    state["state"] = "RUNNING"
    save(state)
    submitted = list(state.get("submitted", []))
    existing = {str(record["key"]) for record in submitted}
    for cell_index, (cell, override) in enumerate(CELLS.items()):
        for seed_index in range(1, 4):
            key = f"{cell}:seed{seed_index}"
            if key in existing:
                continue
            seed = 20310000 + cell_index * 100 + seed_index
            name = f"e3-imagenet100-{cell}-seed{seed_index}"
            run_id = submit(name, seed, override)
            submitted.append({"key": key, "cell": cell, "seed": seed, "override": override, "run_id": run_id})
            existing.add(key)
            state["submitted"] = submitted
            save(state)
    wait_all([str(record["run_id"]) for record in submitted])
    state["state"] = "SUCCEEDED"
    save(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
