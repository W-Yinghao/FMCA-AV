#!/usr/bin/env python3
"""Run the bounded post-fix E4 architecture and permutation-control wave."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION
from scripts.e4_priority_designs import DESIGNS, override


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG = "configs/ssl/cifar10_reference.json"
SEEDS = (20280001, 20280002, 20280003)
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"}


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(path)


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def run_state(run_id: str) -> str:
    return str(read(Path("runs") / run_id / "status.json")["state"])


def wait_all(run_ids: list[str], label: str) -> None:
    while True:
        time.sleep(POLL_SECONDS); refresh()
        states = {run_id: run_state(run_id) for run_id in run_ids}
        failures = {run_id: value for run_id, value in states.items() if value in TERMINAL and value != "SUCCEEDED"}
        if failures:
            raise RuntimeError(f"{label} failed: " + json.dumps(failures, sort_keys=True))
        if all(value == "SUCCEEDED" for value in states.values()): return


def submit(name: str, command: list[str]) -> str:
    argv = ["python3", "-m", "harness.cli", "submit", "--name", name,
            "--gpus", "1", "--profile", "v100", "--", *command]
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def checkpoint(run_id: str) -> str:
    result = read(Path("runs") / run_id / "artifacts" / "train_result.json")
    if result.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing pre-fix E4 source {run_id}")
    value = result.get("last_checkpoint") or result.get("best_checkpoint")
    if not value or not Path(str(value)).is_file(): raise RuntimeError(f"missing E4 checkpoint {run_id}")
    return str(value)


def train_command(design: str, seed_index: int) -> list[str]:
    return [PYTHON, "-m", "scripts.run_fmca_pipeline", "--config", CONFIG,
            "--seed", str(SEEDS[seed_index - 1]), "--overrides-json",
            json.dumps(override(design, seed_index), separators=(",", ":")), "--train-only"]


def probe_command(design: str, seed_index: int, source: str) -> list[str]:
    value = override(design, seed_index); value["probe"] = {
        "max_epochs": 100, "devices": 1, "accelerator": "gpu",
        "limit_train_batches": 1.0, "limit_val_batches": 1.0, "limit_test_batches": 1.0,
    }
    return ["env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(value, separators=(",", ":")),
            f"FMCA_SEED_OVERRIDE={SEEDS[seed_index - 1]}", PYTHON, "-m", "fmca_av.cli",
            "linear-probe", "--config", CONFIG, "--checkpoint", source]


def permutation_command(design: str, seed_index: int, source: str) -> list[str]:
    return [PYTHON, "-m", "scripts.evaluate_view_permutation", "--config", CONFIG,
            "--checkpoint", source, "--overrides-json",
            json.dumps(override(design, seed_index), separators=(",", ":")), "--batches", "8"]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--state-file", required=True)
    parser.add_argument("--validation-run", required=True); args = parser.parse_args()
    path = Path(args.state_file)
    state = read(path) if path.is_file() else {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "state": "RUNNING", "chain_runs": [], "records": [],
    }
    if state.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing legacy E4 priority state {path}")
    chain = list(state.get("chain_runs", [])); current = os.environ["FMCA_HARNESS_RUN_ID"]
    if current not in chain: chain.append(current)
    state["chain_runs"] = chain; state["state"] = "RUNNING"; state["validation_run"] = args.validation_run; write(path, state)
    wait_all([args.validation_run], "E4 parameter/forward validation")
    records = list(state.get("records", [])); existing = {(str(r["design"]), int(r["seed_index"])) for r in records}
    for design in DESIGNS:
        for seed_index in range(1, 4):
            if (design, seed_index) in existing: continue
            run_id = submit(f"priority-e4-cifar10-{design}-seed{seed_index}-epoch200",
                            train_command(design, seed_index))
            records.append({"design": design, "seed_index": seed_index, "seed": SEEDS[seed_index - 1],
                            "train_run": run_id}); existing.add((design, seed_index))
            state["records"] = records; write(path, state)
    wait_all([str(record["train_run"]) for record in records], "E4 training")
    for field, suffix, builder in (
        ("probe_run", "linear-probe", probe_command),
        ("permutation_run", "view-permutation", permutation_command),
    ):
        submitted = []
        for record in records:
            if record.get(field): submitted.append(str(record[field])); continue
            design = str(record["design"]); seed_index = int(record["seed_index"])
            source = checkpoint(str(record["train_run"]))
            run_id = submit(f"priority-e4-cifar10-{design}-seed{seed_index}-{suffix}",
                            builder(design, seed_index, source))
            record[field] = run_id; state["records"] = records; write(path, state); submitted.append(run_id)
        wait_all(submitted, f"E4 {suffix}")
    state["state"] = "SUCCEEDED"; write(path, state); return 0


if __name__ == "__main__":
    raise SystemExit(main())
