#!/usr/bin/env python3
"""Finish the three selected CIFAR-10 FMCA-AV M=8 chains end-to-end."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION
from scripts import formal_ssl_state_machine as formal


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"}
SEEDS = (20280001, 20280002, 20280003)


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def run_state(run_id: str) -> str:
    return str(read(Path("runs") / run_id / "status.json")["state"])


def checkpoint(run_id: str) -> str:
    payload = read(Path("runs") / run_id / "artifacts" / "train_result.json")
    if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing pre-fix continuation source {run_id}")
    value = payload.get("last_checkpoint") or payload.get("best_checkpoint")
    if not value or not Path(str(value)).is_file():
        raise RuntimeError(f"missing checkpoint for {run_id}")
    return str(value)


def wait_all(run_ids: list[str], label: str) -> None:
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        states = {run_id: run_state(run_id) for run_id in run_ids}
        failures = {run_id: value for run_id, value in states.items() if value in TERMINAL and value != "SUCCEEDED"}
        if failures:
            raise RuntimeError(f"{label} failed: " + json.dumps(failures, sort_keys=True))
        if all(value == "SUCCEEDED" for value in states.values()):
            return


def submit(name: str, command: list[str]) -> str:
    argv = ["python3", "-m", "harness.cli", "submit", "--name", name,
            "--gpus", "1", "--profile", "v100", "--", *command]
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def action(seed_index: int, kind: str, target: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "dataset": "cifar10", "method": "fmca_av", "views": 8,
        "seed": SEEDS[seed_index - 1], "seed_index": seed_index,
        "key": f"cifar10:8:fmca_av:{SEEDS[seed_index - 1]}", "kind": kind,
    }
    if target is not None:
        value["target"] = target
    return value


def command(value: dict[str, object], source_checkpoint: str) -> list[str]:
    state = {"last_checkpoints": {str(value["key"]): source_checkpoint}}
    _, _, _, result = formal.command_for(value, state)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--initial-runs", required=True,
                        help="comma-separated seed1,seed2,seed3 epoch-400 run IDs")
    args = parser.parse_args()
    path = Path(args.state_file)
    state = read(path) if path.is_file() else {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "state": "RUNNING", "chain_runs": [], "seeds": {},
    }
    if state.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing legacy priority continuation state {path}")
    chain = list(state.get("chain_runs", [])); current = os.environ["FMCA_HARNESS_RUN_ID"]
    if current not in chain:
        chain.append(current)
    state["chain_runs"] = chain; state["state"] = "RUNNING"
    records = dict(state.get("seeds", {}))
    initial = [value for value in args.initial_runs.split(",") if value]
    if len(initial) != 3:
        raise ValueError("--initial-runs must contain exactly three run IDs")
    for seed_index, run_id in enumerate(initial, 1):
        records.setdefault(str(seed_index), {"target": 400, "train_run": run_id, "history": [run_id]})
    state["seeds"] = records; write(path, state)

    for target in (400, 600, 800):
        current_runs = [str(dict(records[str(index)])["train_run"]) for index in range(1, 4)]
        wait_all(current_runs, f"FMCA M=8 epoch {target}")
        if target == 800:
            break
        for seed_index in range(1, 4):
            record = dict(records[str(seed_index)])
            source = checkpoint(str(record["train_run"]))
            next_target = target + 200
            value = action(seed_index, "train", next_target)
            run_id = submit(
                f"priority-e5-cifar10-fmca-av-v8-seed{seed_index}-epoch{next_target}",
                command(value, source),
            )
            record["target"] = next_target; record["train_run"] = run_id
            record["history"] = [*list(record.get("history", [])), run_id]
            records[str(seed_index)] = record; state["seeds"] = records; write(path, state)

    for kind in ("probe", "knn"):
        field = f"{kind}_run"
        submitted = []
        for seed_index in range(1, 4):
            record = dict(records[str(seed_index)])
            if record.get(field):
                submitted.append(str(record[field])); continue
            source = checkpoint(str(record["train_run"]))
            value = action(seed_index, kind)
            run_id = submit(
                f"priority-e5-cifar10-fmca-av-v8-seed{seed_index}-{kind}-epoch800",
                command(value, source),
            )
            record[field] = run_id; records[str(seed_index)] = record
            state["seeds"] = records; write(path, state); submitted.append(run_id)
        wait_all(submitted, f"FMCA M=8 {kind}")
    state["state"] = "SUCCEEDED"; write(path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
