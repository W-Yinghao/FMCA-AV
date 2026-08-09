#!/usr/bin/env python3
"""Restartable one-action-per-Slurm-job ImageNet formal experiment chain."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

try:
    from scripts.orchestration_retries import MAX_INFRASTRUCTURE_ATTEMPTS, is_infrastructure_failure
except ModuleNotFoundError:
    from orchestration_retries import MAX_INFRASTRUCTURE_ATTEMPTS, is_infrastructure_failure

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
TORCHRUN = "/home/infres/yinwang/FMCA-AV/scripts/torchrun"
REFERENCE = "configs/ssl/imagenet1k_reference.json"
DEFAULT_DEPENDENCY_STATE = "results/orchestration/formal_ssl_postfix_state.json"
FMCA_SEEDS = (20267001, 20267002, 20267003)
FMCA_VARIANTS = ("fmca_av", "fmca_av_matched_head", "hfmca_style", "regular_fmca")
BASELINES = ("simclr", "vicreg", "moco_v2", "dino", "dcca", "vamp2")
BASELINE_SEEDS = FMCA_SEEDS


def actions() -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    # E10 owns the post-fix one-/two-rank CIFAR scaling points.  This ImageNet
    # chain starts directly with ImageNet work and does not duplicate them.
    for variant in FMCA_VARIANTS:
        views = 1 if variant == "regular_fmca" else 8
        for base_seed in FMCA_SEEDS:
            seed = base_seed
            key = f"{variant}:{seed}"
            for target in range(25, 101, 25):
                values.append({"kind": "fmca_train", "key": key, "seed": seed, "target": target,
                               "variant": variant, "views": views})
            values.extend((
                {"kind": "fmca_probe", "key": key, "seed": seed, "variant": variant, "views": views},
                {"kind": "fmca_knn", "key": key, "seed": seed, "variant": variant, "views": views},
                {"kind": "fmca_robustness", "key": key, "seed": seed, "variant": variant, "views": views},
            ))
    for method in BASELINES:
        for seed_index, base_seed in enumerate(BASELINE_SEEDS, 1):
            seed = base_seed
            key = f"baseline:{method}:{seed}"
            for target in range(25, 101, 25):
                values.append({"kind": "baseline_train", "key": key, "method": method, "seed": seed,
                               "seed_index": seed_index, "target": target})
            values.extend((
                {"kind": "baseline_probe", "key": key, "method": method, "seed": seed, "seed_index": seed_index},
                {"kind": "baseline_knn", "key": key, "method": method, "seed": seed, "seed_index": seed_index},
                {"kind": "baseline_robustness", "key": key, "method": method, "seed": seed, "seed_index": seed_index},
            ))
    return values


def dependency_state_value(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing legacy formal SSL dependency state: {path}")
    return str(payload.get("state", "MISSING"))


def load_state(path: Path, dependency_state: str) -> dict[str, object]:
    dependency_path = Path(dependency_state).resolve()
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            raise RuntimeError(f"refusing legacy ImageNet state: {path}")
        completed = list(state.get("completed", []))
        migrated = [
            record for record in completed
            if str(dict(record.get("action", {})).get("kind", "")) == "ddp"
        ]
        if migrated:
            if state.get("current_run") or str(dict(state.get("current_action") or {}).get("kind", "")) == "ddp":
                raise RuntimeError("cannot migrate ImageNet DDP prefix while a child is active")
            state["action_index"] = max(0, int(state.get("action_index", 0)) - len(migrated))
            state["completed"] = [record for record in completed if record not in migrated]
            moved = list(state.get("migrated_e10_ddp_runs", []))
            for record in migrated:
                run_id = str(record.get("run_id", ""))
                if run_id and run_id not in moved:
                    moved.append(run_id)
            state["migrated_e10_ddp_runs"] = moved
        state["dependency_state"] = str(dependency_path)
        state["dependency_complete"] = dependency_state_value(dependency_path) == "SUCCEEDED"
        return state
    return {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "dependency_state": str(dependency_path), "dependency_complete": False,
        "action_index": 0, "current_run": "", "current_action": None,
        "current_attempt": 0, "current_infrastructure_attempt": 0, "current_retry_from": "",
        "last_checkpoints": {}, "probe_checkpoints": {}, "final_train_runs": {},
        "completed": [], "chain_runs": [], "state": "RUNNING",
    }


def save_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def run_state(run_id: str) -> str:
    return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def wait_terminal(run_id: str) -> str:
    while True:
        refresh()
        value = run_state(run_id)
        if value in {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"}:
            return value
        time.sleep(POLL_SECONDS)


def wait_success(run_id: str) -> None:
    value = wait_terminal(run_id)
    if value != "SUCCEEDED":
        raise RuntimeError(f"prerequisite {run_id} ended in {value}")


def wait_dependency_state(path: Path) -> None:
    while True:
        value = dependency_state_value(path)
        if value == "SUCCEEDED":
            return
        if value in {"FAILED", "STOPPED", "BLOCKED"}:
            raise RuntimeError(f"formal SSL dependency state ended in {value}: {path}")
        time.sleep(POLL_SECONDS)
        refresh()


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr:
            raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS)
        refresh()


def last_checkpoint(run_id: str) -> str:
    result_path = Path("runs") / run_id / "artifacts" / "train_result.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            raise RuntimeError(f"refusing legacy ImageNet checkpoint from {run_id}")
        for field in ("last_checkpoint", "best_checkpoint"):
            value = payload.get(field)
            if value and Path(str(value)).is_file():
                return str(value)
    fallback = Path("runs") / run_id / "artifacts" / "checkpoints" / "last.ckpt"
    return str(fallback) if fallback.is_file() else ""


def command_for(action: dict[str, object], state: dict[str, object]) -> tuple[str, int, list[str]]:
    kind = str(action["kind"])
    checkpoints = dict(state["last_checkpoints"])
    key = str(action["key"]); seed = int(action["seed"])
    checkpoint = str(checkpoints.get(key, ""))
    if kind == "fmca_train":
        target = int(action["target"]); variant = str(action["variant"])
        override: dict[str, object] = {
            "experiment": {"name": f"imagenet1k-{variant}-reference"},
            "data": {"num_views": int(action["views"])},
            "trainer": {"max_epochs": target, "checkpoint_save_top_k": 0 if target < 100 else 1},
            "optimizer": {"scheduler_t_max": 100},
        }
        if variant == "fmca_av": override["model"] = {"parent_aggregation": "mean"}
        elif variant == "fmca_av_matched_head": override["model"] = {"parent_aggregation": "mean", "f_head_hidden_dims": [3641, 2048]}
        elif variant == "hfmca_style": override["model"] = {"parent_aggregation": "concat"}
        elif variant == "regular_fmca":
            override["model"] = {"parent_aggregation": "raw"}
            override["data"] = {"num_views": int(action["views"]), "include_raw_parent": True}
        command = [TORCHRUN, "--standalone", "--nnodes=1", "--nproc_per_node=2", "-m", "scripts.run_fmca_pipeline",
                   "--config", REFERENCE, "--seed", str(seed), "--overrides-json", json.dumps(override, separators=(",", ":"))]
        if checkpoint: command += ["--resume", checkpoint]
        if target < 100: command.append("--train-only")
        return f"imagenet1k-{variant}-seed-{seed}-epoch-{target}", 2, command
    if kind == "fmca_probe":
        variant = str(action["variant"])
        override_value: dict[str, object] = {
            "experiment": {"name": f"imagenet1k-{variant}-reference"},
            "data": {"num_views": int(action["views"])}, "probe": {"devices": 2, "accelerator": "gpu"},
        }
        if variant == "fmca_av": override_value["model"] = {"parent_aggregation": "mean"}
        elif variant == "fmca_av_matched_head": override_value["model"] = {"parent_aggregation": "mean", "f_head_hidden_dims": [3641, 2048]}
        elif variant == "hfmca_style": override_value["model"] = {"parent_aggregation": "concat"}
        elif variant == "regular_fmca": override_value["model"] = {"parent_aggregation": "raw"}
        override = json.dumps(override_value, separators=(",", ":"))
        return f"imagenet1k-{variant}-linear-probe-seed-{seed}", 2, [
            TORCHRUN, "--standalone", "--nnodes=1", "--nproc_per_node=2", "-m", "fmca_av.cli",
            "linear-probe", "--config", REFERENCE, "--checkpoint", checkpoint,
            "--seed", str(seed), "--overrides-json", override,
        ]
    if kind == "fmca_knn":
        variant = str(action["variant"])
        override_value = {"experiment": {"name": f"imagenet1k-{variant}-reference"},
                          "data": {"num_views": int(action["views"])}}
        if variant == "fmca_av": override_value["model"] = {"parent_aggregation": "mean"}
        elif variant == "fmca_av_matched_head": override_value["model"] = {"parent_aggregation": "mean", "f_head_hidden_dims": [3641, 2048]}
        elif variant == "hfmca_style": override_value["model"] = {"parent_aggregation": "concat"}
        elif variant == "regular_fmca": override_value["model"] = {"parent_aggregation": "raw"}
        return f"imagenet1k-{variant}-knn-seed-{seed}", 1, [
            PYTHON, "-m", "fmca_av.cli", "knn", "--config", REFERENCE, "--checkpoint", checkpoint,
            "--seed", str(seed), "--overrides-json", json.dumps(override_value, separators=(",", ":")),
            "--workers", "12", "--batch-size", "256", "--bank-chunk-size", "8192",
        ]
    if kind == "fmca_robustness":
        variant = str(action["variant"])
        override_value = {"experiment": {"name": f"imagenet1k-{variant}-reference"},
                          "data": {"num_views": int(action["views"])}}
        if variant == "fmca_av": override_value["model"] = {"parent_aggregation": "mean"}
        elif variant == "fmca_av_matched_head": override_value["model"] = {"parent_aggregation": "mean", "f_head_hidden_dims": [3641, 2048]}
        elif variant == "hfmca_style": override_value["model"] = {"parent_aggregation": "concat"}
        elif variant == "regular_fmca": override_value["model"] = {"parent_aggregation": "raw"}
        probe_checkpoint = str(dict(state["probe_checkpoints"]).get(key, ""))
        return f"imagenet1k-{variant}-robustness-seed-{seed}", 1, [
            PYTHON, "-m", "fmca_av.cli", "imagenet-robustness", "--config", REFERENCE,
            "--checkpoint", checkpoint, "--probe-checkpoint", probe_checkpoint,
            "--root", "/projects/EEG-foundation-model/yinghao/FMCA-AV/robustness", "--suite", "all",
            "--batch-size", "128", "--workers", "12", "--seed", str(seed),
            "--overrides-json", json.dumps(override_value, separators=(",", ":")),
        ]
    method = str(action["method"]); seed_index = int(action["seed_index"])
    base_override: dict[str, object] = {
        "experiment": {"name": f"imagenet1k-{method}-reference", "method": method},
        "data": {"num_views": 8},
    }
    if kind == "baseline_train":
        target = int(action["target"])
        override = {**base_override, "trainer": {"max_epochs": target, "checkpoint_save_top_k": 0 if target < 100 else 1}, "optimizer": {"scheduler_t_max": 100}}
        command = [TORCHRUN, "--standalone", "--nnodes=1", "--nproc_per_node=2", "-m", "fmca_av.baseline_cli", "train",
                   "--config", REFERENCE, "--seed", str(seed), "--overrides-json", json.dumps(override, separators=(",", ":"))]
        if checkpoint: command += ["--resume", checkpoint]
        return f"imagenet1k-{method}-seed{seed_index}-epoch-{target}", 2, command
    if kind == "baseline_probe":
        override = {**base_override, "probe": {"devices": 2, "accelerator": "gpu"}}
        return f"imagenet1k-{method}-linear-probe-seed{seed_index}", 2, [
            TORCHRUN, "--standalone", "--nnodes=1", "--nproc_per_node=2", "-m", "fmca_av.baseline_cli",
            "linear-probe", "--config", REFERENCE, "--checkpoint", checkpoint,
            "--seed", str(seed), "--overrides-json", json.dumps(override, separators=(",", ":")),
        ]
    if kind == "baseline_knn":
        return f"imagenet1k-{method}-knn-seed{seed_index}", 1, [
            PYTHON, "-m", "fmca_av.cli", "knn", "--config", REFERENCE, "--checkpoint", checkpoint,
            "--seed", str(seed), "--overrides-json", json.dumps(base_override, separators=(",", ":")),
            "--workers", "12", "--batch-size", "256", "--bank-chunk-size", "8192",
        ]
    if kind == "baseline_robustness":
        probe_checkpoint = str(dict(state["probe_checkpoints"]).get(key, ""))
        return f"imagenet1k-{method}-robustness-seed{seed_index}", 1, [
            PYTHON, "-m", "fmca_av.cli", "imagenet-robustness", "--config", REFERENCE,
            "--checkpoint", checkpoint, "--probe-checkpoint", probe_checkpoint,
            "--root", "/projects/EEG-foundation-model/yinghao/FMCA-AV/robustness", "--suite", "all",
            "--batch-size", "128", "--workers", "12", "--seed", str(seed),
            "--overrides-json", json.dumps(base_override, separators=(",", ":")),
        ]
    raise ValueError(f"unknown formal action {kind}")


def submit_action(action: dict[str, object], state: dict[str, object], retry_from: str = "") -> str:
    name, gpus, command = command_for(action, state)
    if str(action["kind"]) == "ddp":
        # E10 now compares only the permitted one- and two-rank points.
        profile = ["--profile", "l40s"]
    elif gpus > 1:
        # Two H100 ranks with batch 64/rank preserve the former global batch
        # of 128 while respecting the project-wide two-GPU ceiling.
        profile = ["--profile", "imagenet_ddp"]
    else:
        profile = ["--profile", "imagenet"] if "imagenet1k" in name else []
    retry_args = ["--retry-from", retry_from] if retry_from else []
    return submit(["python3", "-m", "harness.cli", "submit", "--name", name,
                   "--gpus", str(gpus), *profile, *retry_args, "--", *command])


def submit_successor(state_file: Path, dependency_state: str, index: int) -> str:
    return submit([
        "python3", "-m", "harness.cli", "watch", "--name", f"imagenet-formal-chain-step-{index:03d}", "--",
        PYTHON, "-m", "scripts.formal_imagenet_state_machine", "--state-file", str(state_file), "--dependency-state", dependency_state,
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", default=f"results/orchestration/imagenet_formal_{SCIENTIFIC_CORRECTNESS_VERSION}.json")
    parser.add_argument("--dependency", default="", help=argparse.SUPPRESS)
    parser.add_argument("--dependency-state", default=DEFAULT_DEPENDENCY_STATE)
    args = parser.parse_args()
    state_file = Path(args.state_file).resolve()
    state = load_state(state_file, args.dependency_state)
    chain_runs = list(state["chain_runs"]); current_chain_run = os.environ["FMCA_HARNESS_RUN_ID"]
    if current_chain_run not in chain_runs: chain_runs.append(current_chain_run)
    state["chain_runs"] = chain_runs
    save_state(state_file, state)
    if not bool(state.get("dependency_complete", False)):
        wait_dependency_state(Path(str(state["dependency_state"])))
        state["dependency_complete"] = True
        save_state(state_file, state)
    current_run = str(state.get("current_run", ""))
    if current_run:
        terminal = wait_terminal(current_run)
        action = dict(state["current_action"])
        attempt = int(state.get("current_attempt", 1))
        infrastructure_attempt = int(state.get("current_infrastructure_attempt", 0))
        deferred = dict(state.get("deferred", {}))
        operator_deferred = (
            terminal == "STOPPED"
            and str(action.get("kind")) == "ddp"
            and int(action.get("gpus", 0)) == 4
            and str(deferred.get("run_id", "")) == current_run
        )
        if operator_deferred:
            deferred_actions = list(state.get("deferred_actions", []))
            if not any(str(item.get("run_id", "")) == current_run for item in deferred_actions):
                deferred_actions.append({
                    "action": action,
                    "run_id": current_run,
                    "state": "DEFERRED",
                    "reason": str(deferred.get("reason", "operator deferred")),
                })
            state["deferred_actions"] = deferred_actions
            # The four-rank point was removed by operator policy.  Keep
            # action_index=2 so the next successor starts the first ImageNet
            # scientific action in the revised plan.
            state["current_attempt"] = 0
            state["current_infrastructure_attempt"] = 0
            state["current_retry_from"] = ""
        elif terminal == "SUCCEEDED":
            if str(action["kind"]) in {"fmca_train", "baseline_train"}:
                candidate = last_checkpoint(current_run)
                if not candidate: raise RuntimeError(f"successful training action {current_run} has no checkpoint")
                checkpoints = dict(state["last_checkpoints"]); checkpoints[str(action["key"])] = candidate; state["last_checkpoints"] = checkpoints
                if int(action["target"]) == 100:
                    final_runs = dict(state["final_train_runs"]); final_runs[str(action["key"])] = current_run; state["final_train_runs"] = final_runs
            if str(action["kind"]) in {"fmca_probe", "baseline_probe"}:
                result_path = Path("runs") / current_run / "artifacts" / "probe_result.json"
                result = json.loads(result_path.read_text(encoding="utf-8")); candidate = result.get("probe_checkpoint")
                if not candidate or not Path(str(candidate)).is_file(): raise RuntimeError(f"successful probe {current_run} has no checkpoint")
                probes = dict(state["probe_checkpoints"]); probes[str(action["key"])] = str(candidate); state["probe_checkpoints"] = probes
            completed = list(state["completed"]); completed.append({"action": action, "run_id": current_run, "state": terminal}); state["completed"] = completed
            state["action_index"] = int(state["action_index"]) + 1
            state["current_attempt"] = 0; state["current_infrastructure_attempt"] = 0
            state["current_retry_from"] = ""
        else:
            candidate = last_checkpoint(current_run) if str(action["kind"]) in {"fmca_train", "baseline_train"} else ""
            if candidate:
                checkpoints = dict(state["last_checkpoints"]); checkpoints[str(action["key"])] = candidate; state["last_checkpoints"] = checkpoints
            infrastructure = is_infrastructure_failure(current_run, terminal)
            if infrastructure:
                infrastructure_attempt += 1
                if infrastructure_attempt >= MAX_INFRASTRUCTURE_ATTEMPTS:
                    state["state"] = "FAILED"; state["failure"] = {
                        "action": action, "run_id": current_run, "terminal": terminal,
                        "failure_kind": "infrastructure", "scientific_attempt": attempt,
                        "infrastructure_attempt": infrastructure_attempt,
                    }
                    save_state(state_file, state); raise RuntimeError(f"formal action exhausted infrastructure retries: {current_run}")
                # The next submission reuses the same scientific attempt number.
                state["current_attempt"] = max(0, attempt - 1)
                state["current_infrastructure_attempt"] = infrastructure_attempt
            elif attempt >= 3:
                state["state"] = "FAILED"; state["failure"] = {
                    "action": action, "run_id": current_run, "terminal": terminal,
                    "failure_kind": "scientific", "scientific_attempt": attempt,
                    "infrastructure_attempt": infrastructure_attempt,
                }
                save_state(state_file, state); raise RuntimeError(f"formal action failed after {attempt} scientific attempts: {current_run} {terminal}")
            state["current_retry_from"] = current_run
        state["current_run"] = ""; state["current_action"] = None; save_state(state_file, state)
    plan = actions(); index = int(state["action_index"])
    if index >= len(plan):
        state["state"] = "SUCCEEDED"; save_state(state_file, state); return 0
    action = plan[index]
    run_id = submit_action(action, state, str(state.get("current_retry_from", "")))
    state["current_run"] = run_id; state["current_action"] = action
    state["current_attempt"] = int(state.get("current_attempt", 0)) + 1
    save_state(state_file, state)
    successor = submit_successor(state_file, args.dependency_state, index)
    state["successor_run"] = successor; save_state(state_file, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
