#!/usr/bin/env python3
"""Probe available 200-epoch checkpoints for the preregistered priority gate."""

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


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_state(run_id: str) -> str:
    return str(read(Path("runs") / run_id / "status.json")["state"])


def checkpoint(run_id: str) -> str:
    payload = read(Path("runs") / run_id / "artifacts" / "train_result.json")
    if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing pre-fix checkpoint {run_id}")
    value = payload.get("last_checkpoint") or payload.get("best_checkpoint")
    if not value or not Path(str(value)).is_file():
        raise RuntimeError(f"checkpoint missing for {run_id}")
    return str(value)


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def submit_with_retry(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def wait_success(run_id: str, label: str) -> None:
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        state = run_state(run_id)
        if state == "SUCCEEDED":
            return
        if state in TERMINAL:
            raise RuntimeError(f"{label} ended in {state}: {run_id}")


def submit(action: dict[str, object], source_run: str) -> str:
    probe_action = {key: value for key, value in action.items() if key != "target"}
    probe_action["kind"] = "probe"
    state = {"last_checkpoints": {str(action["key"]): checkpoint(source_run)}}
    _, _, profile, command = formal.command_for(probe_action, state)
    name = (
        f"priority-e5-{action['dataset']}-{action['method']}-v{action['views']}"
        f"-seed{action['seed_index']}-probe-epoch{action['target']}"
    )
    profile_args = ["--profile", profile] if profile != "default" else []
    argv = ["python3", "-m", "harness.cli", "submit", "--name", name,
            "--gpus", "1", *profile_args, "--", *command]
    return submit_with_retry(argv)


def candidates(formal_state: dict[str, object]) -> list[dict[str, object]]:
    values = []
    for record in list(formal_state.get("completed", [])):
        action = dict(record["action"])
        if (
            record.get("state") == "SUCCEEDED"
            and action.get("kind") == "train"
            and action.get("dataset") == "cifar10"
            and action.get("method") == "fmca_av"
            and int(action.get("views", 0)) in {2, 8}
            and int(action.get("seed_index", 0)) in {1, 2, 3}
            and int(action.get("target", 0)) == 200
        ):
            values.append({"action": action, "source_run": str(record["run_id"])})
    return sorted(values, key=lambda value: (
        int(dict(value["action"])["seed_index"]), int(dict(value["action"])["views"])
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-state", required=True)
    parser.add_argument("--state-file", required=True)
    args = parser.parse_args()
    formal_path = Path(args.formal_state)
    state_path = Path(args.state_file)
    state = read(state_path) if state_path.is_file() else {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "state": "RUNNING", "chain_runs": [], "probes": [],
    }
    if state.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing legacy priority state {state_path}")
    chain = list(state.get("chain_runs", []))
    current = os.environ["FMCA_HARNESS_RUN_ID"]
    if current not in chain:
        chain.append(current)
    state["chain_runs"] = chain
    state["state"] = "RUNNING"
    write(state_path, state)

    validation_run = str(state.get("validation_run", ""))
    if not validation_run:
        validation_run = submit_with_retry([
            "python3", "-m", "harness.cli", "submit", "--name",
            "priority-scheduler-minimal-regression", "--gpus", "0", "--",
            PYTHON, "-m", "unittest", "discover", "-s", "tests", "-p",
            "test_priority_submission.py", "-v",
        ])
        state["validation_run"] = validation_run
        write(state_path, state)
    wait_success(validation_run, "priority scheduler validation")

    formal_state = read(formal_path)
    if formal_state.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing legacy formal state {formal_path}")
    planned = candidates(formal_state)
    if len(planned) != 6:
        raise RuntimeError(f"expected six paired M=2/M=8 probes, found {len(planned)}")
    probes = list(state.get("probes", []))
    existing = {str(record["key"]) for record in probes}
    for candidate in planned:
        action = dict(candidate["action"])
        key = str(action["key"])
        if key in existing:
            continue
        run_id = submit(action, str(candidate["source_run"]))
        probes.append({"key": key, **candidate, "run_id": run_id})
        existing.add(key)
        state["probes"] = probes
        write(state_path, state)

    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        states = {str(record["run_id"]): run_state(str(record["run_id"])) for record in probes}
        failures = {run_id: value for run_id, value in states.items() if value in TERMINAL and value != "SUCCEEDED"}
        if failures:
            state["state"] = "FAILED"
            state["failures"] = failures
            write(state_path, state)
            raise RuntimeError("priority probe failures: " + json.dumps(failures, sort_keys=True))
        if all(value == "SUCCEEDED" for value in states.values()):
            state["state"] = "SUCCEEDED"
            write(state_path, state)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
