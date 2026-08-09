#!/usr/bin/env python3
"""Restartable formal VOC/COCO transfer chain driven by ImageNet checkpoints."""

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
TARGET_STEPS = 90000
CHUNK_STEPS = 30000
FMCA_METHODS = ("fmca_av", "fmca_av_matched_head", "hfmca_style", "regular_fmca")
METHODS = (*FMCA_METHODS, "simclr", "vicreg", "moco_v2", "dino", "dcca", "vamp2")
SEEDS = (20267001, 20267002, 20267003)
TASKS = {
    "voc_detection": {"dataset": "voc", "task": "detection", "root": "/projects/EEG-foundation-model/yinghao/FMCA-AV/voc"},
    "coco_detection": {"dataset": "coco", "task": "detection", "root": "/projects/EEG-foundation-model/yinghao/FMCA-AV/coco"},
    "coco_instance_segmentation": {"dataset": "coco", "task": "instance_segmentation", "root": "/projects/EEG-foundation-model/yinghao/FMCA-AV/coco"},
}


def refresh() -> None: subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def save(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(path)


def load(path: Path, pretrain_state: str) -> dict[str, object]:
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            raise RuntimeError(f"refusing legacy formal transfer state: {path}")
        return state
    return {"scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
            "pretrain_state": str(Path(pretrain_state).resolve()), "pretrain_complete": False, "source_checkpoints": {},
            "action_index": 0, "current_runs": [], "retry_queue": [], "detector_checkpoints": {},
            "completed": [], "chain_runs": [], "state": "RUNNING"}


def checkpoint_key(method: str, seed: int) -> str:
    return f"{method}:{seed}" if method in FMCA_METHODS else f"baseline:{method}:{seed}"


def source_id(method: str, seed_index: int) -> str:
    return f"{method}:seed{seed_index}"


def wait_pretraining(state: dict[str, object]) -> None:
    path = Path(str(state["pretrain_state"]))
    while True:
        time.sleep(POLL_SECONDS); refresh()
        if not path.is_file(): continue
        payload = json.loads(path.read_text(encoding="utf-8")); value = str(payload.get("state", "RUNNING"))
        if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            raise RuntimeError(f"refusing legacy ImageNet pretraining state: {path}")
        if value == "SUCCEEDED":
            checkpoints = dict(payload.get("last_checkpoints", {}))
            expected = [checkpoint_key(method, seed) for method in METHODS for seed in SEEDS]
            missing = [key for key in expected if key not in checkpoints]
            if missing: raise RuntimeError("ImageNet state lacks transfer checkpoints: " + ",".join(missing))
            state["source_checkpoints"] = {
                source_id(method, seed_index): checkpoints[checkpoint_key(method, seed)]
                for method in METHODS for seed_index, seed in enumerate(SEEDS, 1)
            }
            return
        if value == "FAILED": raise RuntimeError("formal ImageNet pretraining state failed")


def actions() -> list[dict[str, object]]:
    experiments = [
        {"method": method, "seed": seed, "seed_index": seed_index,
         "source_id": source_id(method, seed_index), "task_name": task_name,
         "key": f"{method}:seed{seed_index}:{task_name}"}
        for method in METHODS for seed_index, seed in enumerate(SEEDS, 1) for task_name in TASKS
    ]
    values = []
    for target in range(CHUNK_STEPS, TARGET_STEPS + 1, CHUNK_STEPS):
        values.extend({"target": target, **experiment} for experiment in experiments)
    values.extend(
        {"kind": "voc_multilabel", "method": method, "seed": seed, "seed_index": seed_index,
         "source_id": source_id(method, seed_index), "key": f"{method}:seed{seed_index}:voc_multilabel"}
        for method in METHODS for seed_index, seed in enumerate(SEEDS, 1)
    )
    return values


def method_override(method: str) -> dict[str, object]:
    if method == "fmca_av": return {"experiment": {"name": "imagenet1k-fmca-av-reference"}, "data": {"num_views": 8}, "model": {"parent_aggregation": "mean"}}
    if method == "fmca_av_matched_head": return {"experiment": {"name": "imagenet1k-fmca-av-matched-head-reference"}, "data": {"num_views": 8}, "model": {"parent_aggregation": "mean", "f_head_hidden_dims": [3641, 2048]}}
    if method == "hfmca_style": return {"experiment": {"name": "imagenet1k-hfmca-style-reference"}, "data": {"num_views": 8}, "model": {"parent_aggregation": "concat"}}
    if method == "regular_fmca": return {"experiment": {"name": "imagenet1k-regular-fmca-reference"}, "data": {"num_views": 1}, "model": {"parent_aggregation": "raw"}}
    return {"experiment": {"name": f"imagenet1k-{method}-reference", "method": method}, "data": {"num_views": 8}}


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def submit_action(action: dict[str, object], state: dict[str, object], retry_from: str = "") -> str:
    method = str(action["method"]); seed_index = int(action["seed_index"])
    source_checkpoint = str(dict(state["source_checkpoints"])[str(action["source_id"])])
    if str(action.get("kind", "")) == "voc_multilabel":
        model_type = "fmca" if method in {"fmca_av", "fmca_av_matched_head", "hfmca_style", "regular_fmca"} else "baseline"
        command = ["env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(method_override(method), separators=(",", ":")),
                   PYTHON, "-m", "scripts.run_voc_multilabel_probe", "--config", REFERENCE,
                   "--checkpoint", source_checkpoint, "--model-type", model_type,
                   "--root", "/projects/EEG-foundation-model/yinghao/FMCA-AV/voc", "--epochs", "20",
                   "--batch-size", "128", "--workers", "8"]
        retry_args = ["--retry-from", retry_from] if retry_from else []
        return submit(["python3", "-m", "harness.cli", "submit", "--name", f"formal-transfer-{method}-seed{seed_index}-voc2007-multilabel",
                       "--gpus", "1", "--profile", "imagenet", *retry_args, "--", *command])
    task_name = str(action["task_name"]); target = int(action["target"]); spec = TASKS[task_name]
    command = ["env", "FMCA_CONFIG_OVERRIDES=" + json.dumps(method_override(method), separators=(",", ":")),
               PYTHON, "-m", "scripts.run_coco_transfer", "--config", REFERENCE,
               "--checkpoint", source_checkpoint, "--dataset", str(spec["dataset"]),
               "--root", str(spec["root"]), "--task", str(spec["task"]), "--train-images", "200000",
               "--val-images", "10000", "--max-steps", str(target),
               "--seed-offset", str(int(action["seed"]) + target)]
    detector_checkpoint = str(dict(state["detector_checkpoints"]).get(str(action["key"]), ""))
    if detector_checkpoint: command += ["--resume", detector_checkpoint]
    if target < TARGET_STEPS: command.append("--train-only")
    name = f"formal-transfer-{method}-seed{seed_index}-{task_name}-step-{target}"
    retry_args = ["--retry-from", retry_from] if retry_from else []
    return submit(["python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", "1",
                   "--profile", "imagenet", *retry_args, "--", *command])


def submit_successor(state_file: Path, pretrain_state: str, index: int) -> str:
    return submit(["python3", "-m", "harness.cli", "watch", "--name", f"formal-transfer-chain-step-{index:04d}",
        "--", PYTHON, "-m", "scripts.formal_transfer_state_machine", "--state-file", str(state_file),
        "--pretrain-state", pretrain_state])


def run_state(run_id: str) -> str: return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def wait_terminal(run_ids: list[str]) -> dict[str, str]:
    while True:
        time.sleep(POLL_SECONDS); refresh(); states = {run_id: run_state(run_id) for run_id in run_ids}
        if all(value in {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"} for value in states.values()): return states


def detector_checkpoint(run_id: str) -> str:
    path = Path("runs") / run_id / "artifacts" / "detection_train_result.json"
    if not path.is_file(): return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing legacy detector checkpoint from {run_id}")
    value = payload.get("last_checkpoint")
    return str(value) if value and Path(str(value)).is_file() else ""


def take_batch(state: dict[str, object], plan: list[dict[str, object]]) -> list[dict[str, object]]:
    retries = list(state.get("retry_queue", []))
    source = ([{**record, "origin": "retry"} for record in retries] if retries else
              [{"action": action, "attempt": 1, "infrastructure_attempt": 0, "origin": "plan"}
               for action in plan[int(state["action_index"]):]])
    selected = []; keys = set()
    for record in source:
        key = str(record["action"]["key"])
        if key in keys: break
        selected.append(record); keys.add(key)
        if len(selected) == 6: break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--state-file", default=f"results/orchestration/formal_transfer_{SCIENTIFIC_CORRECTNESS_VERSION}.json")
    parser.add_argument("--pretrain-state", default=DEFAULT_PRETRAIN_STATE); args = parser.parse_args()
    state_file = Path(args.state_file).resolve(); state = load(state_file, args.pretrain_state)
    chain = list(state["chain_runs"]); current_chain_run = os.environ["FMCA_HARNESS_RUN_ID"]
    if current_chain_run not in chain: chain.append(current_chain_run)
    state["chain_runs"] = chain; save(state_file, state)
    if not bool(state["pretrain_complete"]): wait_pretraining(state); state["pretrain_complete"] = True; save(state_file, state)
    current = list(state.get("current_runs", []))
    if current:
        terminals = wait_terminal([str(record["run_id"]) for record in current]); checkpoints = dict(state["detector_checkpoints"])
        completed = list(state["completed"]); retries = list(state.get("retry_queue", []))
        for record in current:
            run_id = str(record["run_id"]); action = dict(record["action"]); attempt = int(record["attempt"]); terminal = terminals[run_id]
            infrastructure_attempt = int(record.get("infrastructure_attempt", 0))
            checkpoint = detector_checkpoint(run_id)
            if checkpoint: checkpoints[str(action["key"])] = checkpoint
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
                    }
                    save(state_file, state); raise RuntimeError(f"formal transfer exhausted {kind} retries: {run_id}")
        state["detector_checkpoints"] = checkpoints; state["completed"] = completed; state["retry_queue"] = retries; state["current_runs"] = []; save(state_file, state)
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
        if record["origin"] == "retry": state["retry_queue"] = list(state["retry_queue"])[1:]
        else: state["action_index"] = int(state["action_index"]) + 1
        state["current_runs"] = launched; save(state_file, state)
    successor = submit_successor(state_file, args.pretrain_state, int(state["action_index"])); state["successor_run"] = successor; save(state_file, state); return 0


if __name__ == "__main__": raise SystemExit(main())
