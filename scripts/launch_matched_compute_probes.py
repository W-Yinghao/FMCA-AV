#!/usr/bin/env python3
"""Evaluate V=2 and V=8 checkpoints at an equal encoded-view budget."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from scripts import formal_ssl_state_machine as formal


POLL_SECONDS = 300
FORMAL_STATE = Path("results/orchestration/formal_ssl_state.json")
STATE_PATH = Path("results/orchestration/matched_compute_state.json")


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(payload: dict[str, object]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def run_state(run_id: str) -> str:
    return str(read(Path("runs") / run_id / "status.json")["state"])


def wait_runs(run_ids: list[str], label: str) -> None:
    if not run_ids:
        return
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        states = {run_id: run_state(run_id) for run_id in run_ids}
        failures = {run_id: value for run_id, value in states.items() if value in {"FAILED", "STOPPED", "BLOCKED"}}
        if failures:
            raise RuntimeError(f"{label} failed: " + json.dumps(failures, sort_keys=True))
        if all(value == "SUCCEEDED" for value in states.values()):
            return


def wait_formal() -> dict[str, object]:
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        if not FORMAL_STATE.is_file():
            continue
        payload = read(FORMAL_STATE)
        value = str(payload.get("state", "RUNNING"))
        if value == "SUCCEEDED":
            return payload
        if value in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError(f"formal SSL state ended in {value}")


def submit(name: str, gpus: int, profile: str, command: list[str]) -> str:
    profile_args = ["--profile", profile] if profile == "imagenet" else []
    argv = ["python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", str(gpus), *profile_args, "--", *command]
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def checkpoint(run_id: str) -> str:
    result = read(Path("runs") / run_id / "artifacts" / "train_result.json")
    value = result.get("last_checkpoint") or result.get("best_checkpoint")
    if not value or not Path(str(value)).is_file():
        raise RuntimeError(f"matched-compute checkpoint missing for {run_id}")
    return str(value)


def signature(action: dict[str, object]) -> tuple[object, ...]:
    return (
        action.get("dataset"), action.get("method"), action.get("seed_index"),
        action.get("backbone", ""), action.get("aggregation", ""),
    )


def main() -> int:
    state = read(STATE_PATH) if STATE_PATH.is_file() else {
        "state": "RUNNING", "chain_runs": [], "tiny_continuations": [], "pairs": [],
    }
    chain_runs = list(state.get("chain_runs", []))
    current_chain_run = os.environ["FMCA_HARNESS_RUN_ID"]
    if current_chain_run not in chain_runs: chain_runs.append(current_chain_run)
    state["chain_runs"] = chain_runs
    state["state"] = "RUNNING"
    save(state)
    formal_state = wait_formal()
    completed = list(formal_state["completed"])
    train_records = [record for record in completed if str(dict(record["action"])["kind"]) == "train"]
    probe_records = [record for record in completed if str(dict(record["action"])["kind"]) == "probe"]

    train_lookup = {
        (signature(dict(record["action"])), int(dict(record["action"])["views"]), int(dict(record["action"])["target"])): record
        for record in train_records
    }
    probe_lookup = {
        (signature(dict(record["action"])), int(dict(record["action"])["views"])): record
        for record in probe_records
    }
    v8_actions = []
    for record in train_records:
        action = dict(record["action"])
        if int(action["views"]) != 8:
            continue
        target = int(formal.DATASETS[str(action["dataset"])]["epochs"]) // 4
        if int(action["target"]) == target or (str(action["dataset"]) == "tinyimagenet200" and int(action["target"]) == 40):
            v8_actions.append(action)

    continuations = list(state.get("tiny_continuations", []))
    continuation_by_key = {str(record["key"]): record for record in continuations}
    for action in v8_actions:
        if str(action["dataset"]) != "tinyimagenet200" or int(action["target"]) != 40:
            continue
        key = str(action["key"])
        if key in continuation_by_key:
            continue
        source_record = train_lookup[(signature(action), 8, 40)]
        resumed_action = {**action, "target": 50}
        source_checkpoint = checkpoint(str(source_record["run_id"]))
        run_state_payload = {"last_checkpoints": {key: source_checkpoint}}
        name, gpus, profile, command = formal.command_for(resumed_action, run_state_payload)
        run_id = submit(name + "-matched-compute", gpus, profile, command)
        record = {"key": key, "action": resumed_action, "source_run": source_record["run_id"], "run_id": run_id}
        continuations.append(record)
        continuation_by_key[key] = record
        state["tiny_continuations"] = continuations
        save(state)
    wait_runs([str(record["run_id"]) for record in continuations], "TinyImageNet exact-budget continuations")

    existing_pairs = {str(record["key"]): record for record in list(state.get("pairs", []))}
    pairs = list(state.get("pairs", []))
    for action in v8_actions:
        key = str(action["key"])
        if key in existing_pairs:
            continue
        dataset = str(action["dataset"])
        target = int(formal.DATASETS[dataset]["epochs"]) // 4
        if dataset == "tinyimagenet200" and key in continuation_by_key:
            source_run = str(continuation_by_key[key]["run_id"])
        else:
            source_run = str(train_lookup[(signature(action), 8, target)]["run_id"])
        source_checkpoint = checkpoint(source_run)
        probe_action = {key_name: value for key_name, value in action.items() if key_name != "target"}
        probe_action["kind"] = "probe"
        run_state_payload = {"last_checkpoints": {key: source_checkpoint}}
        name, gpus, profile, command = formal.command_for(probe_action, run_state_payload)
        probe_run = submit(name + "-matched-compute", gpus, profile, command)
        v2_probe = probe_lookup[(signature(action), 2)]
        pair = {
            "key": key, "dataset": dataset, "method": action["method"], "seed_index": action["seed_index"],
            "backbone": action.get("backbone", ""), "aggregation": action.get("aggregation", ""),
            "v2_epochs": int(formal.DATASETS[dataset]["epochs"]), "v8_epochs": target,
            "encoded_view_budget_ratio": 1.0, "v2_probe_run": v2_probe["run_id"],
            "v8_source_run": source_run, "v8_probe_run": probe_run,
        }
        pairs.append(pair)
        existing_pairs[key] = pair
        state["pairs"] = pairs
        save(state)
    wait_runs([str(record["v8_probe_run"]) for record in pairs], "matched-compute probes")
    state["state"] = "SUCCEEDED"
    save(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
