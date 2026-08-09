#!/usr/bin/env python3
"""Restartable ImageNet-1K low-label probes and fine-tuning after formal SSL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

try:
    from scripts.orchestration_retries import is_infrastructure_failure, retry_record
except ModuleNotFoundError:
    from orchestration_retries import is_infrastructure_failure, retry_record

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
REFERENCE = "configs/ssl/imagenet1k_reference.json"
DEFAULT_PRETRAIN_STATE = f"results/orchestration/imagenet_formal_{SCIENTIFIC_CORRECTNESS_VERSION}.json"
FMCA_METHODS = ("fmca_av", "fmca_av_matched_head", "hfmca_style", "regular_fmca")
BASELINES = ("simclr", "vicreg", "moco_v2", "dino")
SEEDS = (20267001, 20267002, 20267003)
PROTOCOLS = (
    ("linear-probe", 0.01), ("linear-probe", 0.1), ("linear-probe", 1.0),
    ("fine-tune", 0.01), ("fine-tune", 0.1),
)


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load(path: Path, pretrain_state: str) -> dict[str, object]:
    if path.is_file():
        state = read(path)
        if state.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            raise RuntimeError(f"refusing legacy ImageNet low-label state: {path}")
        return state
    return {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "pretrain_state": str(Path(pretrain_state).resolve()), "pretrain_complete": False,
        "source_checkpoints": {}, "action_index": 0, "current_runs": [],
        "retry_queue": [], "completed": [], "chain_runs": [], "state": "RUNNING",
    }


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def run_state(run_id: str) -> str:
    return str(read(Path("runs") / run_id / "status.json")["state"])


def source_key(method: str, seed: int) -> str:
    return f"{method}:{seed}" if method in FMCA_METHODS else f"baseline:{method}:{seed}"


def wait_pretraining(state: dict[str, object]) -> None:
    path = Path(str(state["pretrain_state"]))
    expected = [source_key(method, seed) for method in (*FMCA_METHODS, *BASELINES) for seed in SEEDS]
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        if not path.is_file():
            continue
        payload = read(path)
        if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            raise RuntimeError(f"refusing legacy ImageNet pretraining state: {path}")
        value = str(payload.get("state", "RUNNING"))
        if value in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError(f"formal ImageNet pretraining ended in {value}")
        if value != "SUCCEEDED":
            continue
        checkpoints = dict(payload.get("last_checkpoints", {}))
        missing = [key for key in expected if key not in checkpoints or not Path(str(checkpoints[key])).is_file()]
        if missing:
            raise RuntimeError("formal ImageNet state lacks low-label checkpoints: " + ",".join(missing))
        state["source_checkpoints"] = {key: str(checkpoints[key]) for key in expected}
        return


def actions() -> list[dict[str, object]]:
    values = []
    for method in (*FMCA_METHODS, *BASELINES):
        for seed_index, seed in enumerate(SEEDS, 1):
            for protocol, fraction in PROTOCOLS:
                values.append({
                    "method": method, "seed": seed, "seed_index": seed_index,
                    "protocol": protocol, "fraction": fraction,
                    "key": f"{method}:{seed}:{protocol}:{fraction}",
                })
    return values


def method_override(method: str, fraction: float) -> dict[str, object]:
    views = 1 if method == "regular_fmca" else 8
    override: dict[str, object] = {
        "experiment": {"name": f"imagenet1k-{method}-reference"},
        "data": {"num_views": views},
        "probe": {"label_fraction": fraction, "devices": 1, "accelerator": "gpu"},
    }
    if method not in FMCA_METHODS:
        override["experiment"] = {"name": f"imagenet1k-{method}-reference", "method": method}
    elif method == "fmca_av":
        override["model"] = {"parent_aggregation": "mean"}
    elif method == "fmca_av_matched_head":
        override["model"] = {"parent_aggregation": "mean", "f_head_hidden_dims": [3641, 2048]}
    elif method == "hfmca_style":
        override["model"] = {"parent_aggregation": "concat"}
    elif method == "regular_fmca":
        override["model"] = {"parent_aggregation": "raw"}
    return override


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def submit_action(action: dict[str, object], state: dict[str, object], retry_from: str = "") -> str:
    method = str(action["method"]); seed = int(action["seed"])
    protocol = str(action["protocol"]); fraction = float(action["fraction"])
    checkpoint = str(dict(state["source_checkpoints"])[source_key(method, seed)])
    module = "fmca_av.cli" if method in FMCA_METHODS else "fmca_av.baseline_cli"
    tag = str(fraction).replace(".", "p")
    name = f"formal-lowlabel-imagenet1k-{method}-seed{action['seed_index']}-{protocol}-{tag}"
    command = [
        PYTHON, "-m", module, protocol, "--config", REFERENCE,
        "--checkpoint", checkpoint, "--seed", str(seed), "--overrides-json",
        json.dumps(method_override(method, fraction), separators=(",", ":")),
    ]
    retry_args = ["--retry-from", retry_from] if retry_from else []
    return submit([
        "python3", "-m", "harness.cli", "submit", "--name", name,
        "--gpus", "1", "--profile", "imagenet", *retry_args, "--", *command,
    ])


def wait_terminal(run_ids: list[str]) -> dict[str, str]:
    while True:
        time.sleep(POLL_SECONDS)
        refresh()
        states = {run_id: run_state(run_id) for run_id in run_ids}
        if all(value in {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"} for value in states.values()):
            return states


def take_batch(state: dict[str, object], plan: list[dict[str, object]]) -> list[dict[str, object]]:
    retries = list(state.get("retry_queue", []))
    source = ([{**record, "origin": "retry"} for record in retries] if retries else [
        {"action": action, "attempt": 1, "infrastructure_attempt": 0, "origin": "plan"}
        for action in plan[int(state["action_index"]):]
    ])
    return source[:6]


def submit_successor(state_file: Path, pretrain_state: str, index: int) -> str:
    return submit([
        "python3", "-m", "harness.cli", "watch", "--name",
        f"imagenet-lowlabel-chain-step-{index:03d}", "--",
        PYTHON, "-m", "scripts.formal_imagenet_low_label_state_machine",
        "--state-file", str(state_file), "--pretrain-state", pretrain_state,
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", default=f"results/orchestration/formal_imagenet_low_label_{SCIENTIFIC_CORRECTNESS_VERSION}.json")
    parser.add_argument("--pretrain-state", default=DEFAULT_PRETRAIN_STATE)
    args = parser.parse_args(); state_file = Path(args.state_file).resolve()
    state = load(state_file, args.pretrain_state)
    chain = list(state["chain_runs"]); current_chain_run = os.environ["FMCA_HARNESS_RUN_ID"]
    if current_chain_run not in chain: chain.append(current_chain_run)
    state["chain_runs"] = chain; state["state"] = "RUNNING"; save(state_file, state)
    if not bool(state["pretrain_complete"]):
        wait_pretraining(state); state["pretrain_complete"] = True; save(state_file, state)
    current = list(state.get("current_runs", []))
    if current:
        terminals = wait_terminal([str(record["run_id"]) for record in current])
        completed = list(state["completed"]); retries = list(state.get("retry_queue", []))
        for record in current:
            run_id = str(record["run_id"]); action = dict(record["action"])
            attempt = int(record["attempt"]); terminal = terminals[run_id]
            infrastructure_attempt = int(record.get("infrastructure_attempt", 0))
            if terminal == "SUCCEEDED":
                completed.append({"action": action, "run_id": run_id, "state": terminal})
            else:
                retry = retry_record(action, run_id, terminal, attempt, infrastructure_attempt)
                if retry is not None: retries.append(retry)
                else:
                    kind = "infrastructure" if is_infrastructure_failure(run_id, terminal) else "scientific"
                    state["state"] = "FAILED"
                    state["failure"] = {
                        "action": action, "run_id": run_id, "terminal": terminal,
                        "failure_kind": kind, "scientific_attempt": attempt,
                        "infrastructure_attempt": infrastructure_attempt,
                    }
                    save(state_file, state)
                    raise RuntimeError(f"ImageNet low-label action exhausted {kind} retries: {run_id}")
        state["completed"] = completed; state["retry_queue"] = retries
        state["current_runs"] = []; save(state_file, state)
    plan = actions()
    if int(state["action_index"]) >= len(plan) and not state.get("retry_queue"):
        state["state"] = "SUCCEEDED"; save(state_file, state); return 0
    batch = take_batch(state, plan); launched = []
    for record in batch:
        action = dict(record["action"]); run_id = submit_action(action, state, str(record.get("retry_from", "")))
        launched.append({
            "run_id": run_id, "action": action, "attempt": int(record["attempt"]),
            "infrastructure_attempt": int(record.get("infrastructure_attempt", 0)),
        })
        if record["origin"] == "retry":
            state["retry_queue"] = list(state["retry_queue"])[1:]
        else:
            state["action_index"] = int(state["action_index"]) + 1
        state["current_runs"] = launched; save(state_file, state)
    state["successor_run"] = submit_successor(state_file, args.pretrain_state, int(state["action_index"]))
    save(state_file, state); return 0


if __name__ == "__main__":
    raise SystemExit(main())
