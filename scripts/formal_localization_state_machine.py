#!/usr/bin/env python3
"""Formal multi-method localization/faithfulness chain after ImageNet pretraining."""

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


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
REFERENCE = "configs/ssl/imagenet1k_reference.json"
DEFAULT_PRETRAIN_STATE = "results/orchestration/imagenet_formal_state.json"
FMCA_METHODS = ("fmca_av", "fmca_av_matched_head", "hfmca_style", "regular_fmca")
METHODS = (*FMCA_METHODS, "simclr", "vicreg", "moco_v2", "dino", "dcca", "vamp2")
SEEDS = (20267001, 20267002, 20267003)
DATASETS = {
    "cub": ("/projects/EEG-foundation-model/yinghao/FMCA-AV/cub", []),
    "voc": ("/projects/EEG-foundation-model/yinghao/FMCA-AV/voc/VOC2012", []),
    "imagenet": ("/projects/EEG-foundation-model/yinghao/FMCA-AV/imagenet/ILSVRC", ["--labels", "/projects/common/imagenet/LOC_val_solution.csv"]),
}


def refresh() -> None: subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)
def read(path: Path) -> dict[str, object]: return json.loads(path.read_text(encoding="utf-8"))
def save(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(path)


def load(path: Path, pretrain_state: str) -> dict[str, object]:
    if path.is_file(): return read(path)
    return {"pretrain_state": str(Path(pretrain_state).resolve()), "pretrain_complete": False, "source_checkpoints": {},
            "calibrations": {}, "action_index": 0, "current_runs": [], "retry_queue": [], "completed": [],
            "chain_runs": [], "state": "RUNNING"}


def checkpoint_key(method: str, seed: int) -> str:
    return f"{method}:{seed}" if method in FMCA_METHODS else f"baseline:{method}:{seed}"


def source_id(method: str, seed_index: int) -> str:
    return f"{method}:seed{seed_index}"


def wait_pretraining(state: dict[str, object]) -> None:
    path = Path(str(state["pretrain_state"]))
    while True:
        time.sleep(POLL_SECONDS); refresh()
        if not path.is_file(): continue
        payload = read(path); status = str(payload.get("state", "RUNNING"))
        if status == "FAILED": raise RuntimeError("formal ImageNet pretraining state failed")
        if status != "SUCCEEDED": continue
        checkpoints = dict(payload["last_checkpoints"]); final_runs = dict(payload["final_train_runs"])
        expected = [checkpoint_key(method, seed) for method in METHODS for seed in SEEDS]
        missing = [key for key in expected if key not in checkpoints]
        if missing: raise RuntimeError("ImageNet state lacks localization checkpoints: " + ",".join(missing))
        state["source_checkpoints"] = {
            source_id(method, seed_index): checkpoints[checkpoint_key(method, seed)]
            for method in METHODS for seed_index, seed in enumerate(SEEDS, 1)
        }
        state["calibrations"] = {
            source_id(method, seed_index): str(
                Path("runs") / str(final_runs[checkpoint_key(method, seed)]) / "artifacts" / "calibration.pt"
            )
            for method in FMCA_METHODS for seed_index, seed in enumerate(SEEDS, 1)
        }
        return


def override(method: str) -> dict[str, object]:
    if method == "fmca_av": return {"experiment": {"name": "imagenet1k-fmca-av-reference", "method": method}, "data": {"num_views": 8}, "model": {"parent_aggregation": "mean"}}
    if method == "fmca_av_matched_head": return {"experiment": {"name": "imagenet1k-fmca-av-matched-head-reference", "method": method}, "data": {"num_views": 8}, "model": {"parent_aggregation": "mean", "f_head_hidden_dims": [3641, 2048]}}
    if method == "hfmca_style": return {"experiment": {"name": "imagenet1k-hfmca-style-reference", "method": method}, "data": {"num_views": 8}, "model": {"parent_aggregation": "concat"}}
    if method == "regular_fmca": return {"experiment": {"name": "imagenet1k-regular-fmca-reference", "method": method}, "data": {"num_views": 1}, "model": {"parent_aggregation": "raw"}}
    return {"experiment": {"name": f"imagenet1k-{method}-reference", "method": method}, "data": {"num_views": 8}}


def actions() -> list[dict[str, object]]:
    values = []
    for method in METHODS:
        for seed_index, seed in enumerate(SEEDS, 1):
            common = {"method": method, "seed": seed, "seed_index": seed_index,
                      "source_id": source_id(method, seed_index)}
            for dataset in DATASETS:
                values.append({**common, "dataset": dataset, "randomized": False,
                               "key": f"{method}:seed{seed_index}:{dataset}"})
            values.append({**common, "dataset": "cub", "randomized": True,
                           "key": f"{method}:seed{seed_index}:cub:randomized"})
            values.append({**common, "dataset": "cub", "randomized": False, "composition": True,
                           "key": f"{method}:seed{seed_index}:cnn-composition"})
    return values


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def submit_action(action: dict[str, object], state: dict[str, object], retry_from: str = "") -> str:
    method = str(action["method"]); dataset = str(action["dataset"]); randomized = bool(action["randomized"])
    identity = str(action["source_id"]); seed_index = int(action["seed_index"])
    if bool(action.get("composition", False)):
        environment = ["env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override(method), separators=(",", ":"))]
        model_type = "fmca" if method in {"fmca_av", "fmca_av_matched_head", "hfmca_style", "regular_fmca"} else "baseline"
        command = [*environment, PYTHON, "-m", "scripts.run_cnn_composition_maps", "--config", REFERENCE,
                   "--checkpoint", str(dict(state["source_checkpoints"])[identity]), "--model-type", model_type,
                   "--root", DATASETS["cub"][0], "--calibration-samples", "50", "--evaluation-samples", "50"]
        retry_args = ["--retry-from", retry_from] if retry_from else []
        return submit(["python3", "-m", "harness.cli", "submit", "--name", f"formal-e9-{method}-seed{seed_index}-cnn-composition",
                       "--gpus", "1", "--profile", "imagenet", *retry_args, "--", *command])
    root, extra = DATASETS[dataset]; environment = ["env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(override(method), separators=(",", ":"))]
    common = ["--config", REFERENCE, "--checkpoint", str(dict(state["source_checkpoints"])[identity]),
              "--dataset", dataset, "--root", root, "--samples", "100", *extra]
    if method in FMCA_METHODS:
        command = [*environment, PYTHON, "-m", "scripts.run_dependence_localization", *common,
                   "--calibration", str(dict(state["calibrations"])[identity])]
    else: command = [*environment, PYTHON, "-m", "scripts.run_baseline_localization", *common]
    if randomized: command.append("--randomize-backbone")
    name = f"formal-e9-{method}-seed{seed_index}-{dataset}" + ("-randomized" if randomized else "")
    retry_args = ["--retry-from", retry_from] if retry_from else []
    return submit(["python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1",
                   "--profile", "imagenet", *retry_args, "--", *command])


def submit_successor(state_file: Path, pretrain_state: str, index: int) -> str:
    return submit(["python3", "-m", "harness.cli", "watch", "--name", f"formal-e9-chain-step-{index:03d}", "--",
                   PYTHON, "-m", "scripts.formal_localization_state_machine", "--state-file", str(state_file), "--pretrain-state", pretrain_state])


def run_state(run_id: str) -> str: return str(read(Path("runs") / run_id / "status.json")["state"])
def wait_terminal(run_ids: list[str]) -> dict[str, str]:
    while True:
        time.sleep(POLL_SECONDS); refresh(); states = {run_id: run_state(run_id) for run_id in run_ids}
        if all(value in {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"} for value in states.values()): return states


def take_batch(state: dict[str, object], plan: list[dict[str, object]]) -> list[dict[str, object]]:
    retries = list(state.get("retry_queue", []))
    return (([{**record, "origin": "retry"} for record in retries] if retries else
             [{"action": action, "attempt": 1, "infrastructure_attempt": 0, "origin": "plan"}
              for action in plan[int(state["action_index"]):]])[:6])


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--state-file", default="results/orchestration/formal_localization_state.json")
    parser.add_argument("--pretrain-state", default=DEFAULT_PRETRAIN_STATE); args = parser.parse_args(); state_file = Path(args.state_file).resolve()
    state = load(state_file, args.pretrain_state); chain = list(state["chain_runs"]); current_chain_run = os.environ["FMCA_HARNESS_RUN_ID"]
    if current_chain_run not in chain: chain.append(current_chain_run)
    state["chain_runs"] = chain; save(state_file, state)
    if not bool(state["pretrain_complete"]): wait_pretraining(state); state["pretrain_complete"] = True; save(state_file, state)
    current = list(state.get("current_runs", []))
    if current:
        terminals = wait_terminal([str(record["run_id"]) for record in current]); completed = list(state["completed"]); retries = list(state.get("retry_queue", []))
        for record in current:
            run_id = str(record["run_id"]); action = dict(record["action"]); attempt = int(record["attempt"]); terminal = terminals[run_id]
            infrastructure_attempt = int(record.get("infrastructure_attempt", 0))
            if terminal == "SUCCEEDED": completed.append({"action": action, "run_id": run_id, "state": terminal})
            else:
                retry = retry_record(action, run_id, terminal, attempt, infrastructure_attempt)
                if retry is not None: retries.append(retry)
                else:
                    kind = "infrastructure" if is_infrastructure_failure(run_id, terminal) else "scientific"
                    state["state"] = "FAILED"; state["failure"] = {
                        "action": action, "run_id": run_id, "terminal": terminal,
                        "failure_kind": kind, "scientific_attempt": attempt,
                        "infrastructure_attempt": infrastructure_attempt,
                    }; save(state_file, state)
                    raise RuntimeError(f"formal localization exhausted {kind} retries: {run_id}")
        state["completed"] = completed; state["retry_queue"] = retries; state["current_runs"] = []; save(state_file, state)
    plan = actions()
    if int(state["action_index"]) >= len(plan) and not state.get("retry_queue"):
        state["state"] = "SUCCEEDED"; save(state_file, state); return 0
    batch = take_batch(state, plan); launched = []
    for record in batch:
        action = dict(record["action"]); run_id = submit_action(action, state, str(record.get("retry_from", ""))); launched.append({
            "run_id": run_id, "action": action, "attempt": int(record["attempt"]),
            "infrastructure_attempt": int(record.get("infrastructure_attempt", 0)),
        })
        if record["origin"] == "retry": state["retry_queue"] = list(state["retry_queue"])[1:]
        else: state["action_index"] = int(state["action_index"]) + 1
        state["current_runs"] = launched; save(state_file, state)
    successor = submit_successor(state_file, args.pretrain_state, int(state["action_index"])); state["successor_run"] = successor; save(state_file, state); return 0


if __name__ == "__main__": raise SystemExit(main())
