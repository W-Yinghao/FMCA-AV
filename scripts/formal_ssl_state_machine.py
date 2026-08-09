#!/usr/bin/env python3
"""Restartable matched-view confirmatory SSL chain for small/medium datasets."""

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
TORCHRUN = "/home/infres/yinwang/FMCA-AV/scripts/torchrun"
BASELINES = ("simclr", "barlow_twins", "vicreg", "spectral_contrastive", "fastsiam", "byol", "moco_v2", "dino", "dcca", "vamp2")
FMCA_METHODS = {"fmca_av", "fmca_av_matched_head", "fmca_av_deepsets", "hfmca_style", "regular_fmca"}
DEFAULT_DEPENDENCIES = (
    "20260807-062346_launch-e4-aggregation-ablation-wave",
    "20260807-065037_launch-e5-cifar10-matched-views-screening",
    "20260807-060445_launch-cross-dataset-baseline-wave-fixed",
)
DATASETS = {
    "cifar10": {"config": "configs/ssl/cifar10_reference.json", "epochs": 800, "chunk": 200, "seeds": 5, "gpus": 1},
    "cifar100": {"config": "configs/ssl/cifar100_smoke.json", "epochs": 800, "chunk": 200, "seeds": 5, "gpus": 1},
    "stl10": {"config": "configs/ssl/stl10_smoke.json", "epochs": 400, "chunk": 100, "seeds": 5, "gpus": 1},
    "tinyimagenet200": {"config": "configs/ssl/tinyimagenet200_smoke.json", "epochs": 200, "chunk": 50, "seeds": 3, "gpus": 1},
    "imagenet100": {"config": "configs/ssl/imagenet100_smoke.json", "epochs": 100, "chunk": 25, "seeds": 3, "gpus": 2},
}


def experiment_seed(dataset_index: int, seed_index: int) -> int:
    """Keep RNG/data order paired across methods and matched-view protocols."""
    return 20280000 + dataset_index * 10000 + seed_index


def actions() -> list[dict[str, object]]:
    # Keep every ImageNet-derived action behind all non-ImageNet work.  The
    # state machine stores an action index, so preserve the existing CIFAR-10
    # prefix exactly while only moving the still-distant ImageNet blocks.
    non_imagenet_values: list[dict[str, object]] = []
    imagenet_values: list[dict[str, object]] = []
    for dataset_index, (dataset, spec) in enumerate(DATASETS.items()):
        values = imagenet_values if dataset == "imagenet100" else non_imagenet_values
        experiments: list[dict[str, object]] = []
        protocols = [
            ("fmca_av", 2), ("fmca_av", 8),
            ("fmca_av_matched_head", 2), ("fmca_av_matched_head", 8),
            ("fmca_av_deepsets", 2), ("fmca_av_deepsets", 8),
            ("hfmca_style", 2), ("hfmca_style", 8), ("regular_fmca", 1),
        ]
        protocols.extend((method, views) for method in BASELINES for views in (2, 8))
        for method, views in protocols:
            for seed_index in range(1, int(spec["seeds"]) + 1):
                seed = experiment_seed(dataset_index, seed_index)
                key = f"{dataset}:{views}:{method}:{seed}"
                experiments.append({"dataset": dataset, "views": views, "method": method,
                                    "seed": seed, "seed_index": seed_index, "key": key})
        # Stage-major ordering keeps each checkpoint chain sequential while allowing
        # independent one-GPU chains to share the configured server capacity.
        for target in range(int(spec["chunk"]), int(spec["epochs"]) + 1, int(spec["chunk"])):
            values.extend({"kind": "train", "target": target, **experiment} for experiment in experiments)
        values.extend({"kind": "probe", **experiment} for experiment in experiments)
        values.extend({"kind": "knn", **experiment} for experiment in experiments)
    for dataset_index, dataset in enumerate(("cifar100", "imagenet100"), 1):
        values = imagenet_values if dataset == "imagenet100" else non_imagenet_values
        spec = DATASETS[dataset]
        experiments = []
        for backbone_index, backbone in enumerate(("convnext_tiny", "vit_s_16")):
            for aggregation_index, aggregation in enumerate(("mean", "concat")):
                for views in (2, 8):
                    for seed_index in range(1, 4):
                        seed = 20295000 + dataset_index * 10000 + seed_index
                        key = f"architecture:{dataset}:{backbone}:{aggregation}:{views}:{seed}"
                        common = {"dataset": dataset, "views": views, "method": "fmca_av", "seed": seed,
                                  "seed_index": seed_index, "key": key, "backbone": backbone, "aggregation": aggregation}
                        experiments.append(common)
        for target in range(int(spec["chunk"]), int(spec["epochs"]) + 1, int(spec["chunk"])):
            values.extend({"kind": "train", "target": target, **experiment} for experiment in experiments)
        values.extend({"kind": "probe", **experiment} for experiment in experiments)
        values.extend({"kind": "knn", **experiment} for experiment in experiments)
    return non_imagenet_values + imagenet_values


def load_state(path: Path, dependencies: list[str]) -> dict[str, object]:
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        recorded = state.get("scientific_correctness_version")
        if recorded != SCIENTIFIC_CORRECTNESS_VERSION:
            raise RuntimeError(
                f"refusing legacy formal SSL state {path}: correctness version "
                f"{recorded!r} != {SCIENTIFIC_CORRECTNESS_VERSION!r}"
            )
        return state
    return {"dependencies": dependencies, "dependencies_complete": False, "action_index": 0,
            "current_runs": [], "retry_queue": [], "last_checkpoints": {},
            "completed": [], "chain_runs": [], "state": "RUNNING",
            "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION}


def save_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(path)


def refresh() -> None: subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)
def run_state(run_id: str) -> str: return str(json.loads((Path("runs") / run_id / "status.json").read_text(encoding="utf-8"))["state"])


def latest_retry(run_id: str) -> str:
    jobs = dict(json.loads(Path("harness/state/jobs.json").read_text(encoding="utf-8")).get("jobs", {}))
    current = run_id; visited = {current}
    while True:
        children = [value for value in jobs.values()
                    if str(value.get("retry_from", "")) == current
                    and str(value.get("run_id", "")) not in visited]
        if not children: return current
        chosen = max(children, key=lambda value: str(value.get("created_at", "")))
        current = str(chosen["run_id"]); visited.add(current)


def wait_terminal(run_ids: list[str]) -> dict[str, str]:
    while True:
        time.sleep(POLL_SECONDS); refresh()
        resolved = [latest_retry(run_id) for run_id in run_ids]
        states = {run_id: run_state(run_id) for run_id in resolved}
        if all(value in {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"} for value in states.values()): return states


def wait_for_progress(run_ids: list[str]) -> dict[str, str]:
    """Return after at least one member of an in-flight batch becomes terminal."""
    terminal_states = {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"}
    while True:
        states = {run_id: run_state(run_id) for run_id in run_ids}
        if any(value in terminal_states for value in states.values()):
            return states
        time.sleep(POLL_SECONDS); refresh()


def wait_dependencies(run_ids: list[str]) -> None:
    states = wait_terminal(run_ids); failures = {key: value for key, value in states.items() if value != "SUCCEEDED"}
    if failures: raise RuntimeError("confirmatory dependencies failed: " + json.dumps(failures, sort_keys=True))


def submit(argv: list[str]) -> str:
    while True:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0: return result.stdout.strip()
        if "GPU limit exceeded" not in result.stderr: raise RuntimeError(result.stderr.strip())
        time.sleep(POLL_SECONDS); refresh()


def last_checkpoint(run_id: str) -> str:
    result_path = Path("runs") / run_id / "artifacts" / "train_result.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        recorded = payload.get("scientific_correctness_version")
        if recorded != SCIENTIFIC_CORRECTNESS_VERSION:
            raise RuntimeError(
                f"refusing pre-fix checkpoint from {run_id}: correctness version "
                f"{recorded!r} != {SCIENTIFIC_CORRECTNESS_VERSION!r}"
            )
        for field in ("last_checkpoint", "best_checkpoint"):
            value = payload.get(field)
            if value and Path(str(value)).is_file(): return str(value)
    fallback = Path("runs") / run_id / "artifacts" / "checkpoints" / "last.ckpt"
    return str(fallback) if fallback.is_file() else ""


def training_override(action: dict[str, object]) -> dict[str, object]:
    dataset = str(action["dataset"]); method = str(action["method"]); spec = DATASETS[dataset]
    override: dict[str, object] = {
        "experiment": {"name": f"{dataset}-{method}-v{action['views']}-confirmatory"},
        "data": {"num_views": int(action["views"])},
        "trainer": {
            "max_epochs": int(action["target"]),
            "checkpoint_save_top_k": 0 if int(action["target"]) < int(spec["epochs"]) else 1,
        },
        "optimizer": {"scheduler_t_max": int(spec["epochs"])},
    }
    if method not in FMCA_METHODS: override["experiment"] = {"name": f"{dataset}-{method}-v{action['views']}-confirmatory", "method": method}
    model_override: dict[str, object] = {}
    if method == "fmca_av_deepsets":
        model_override["parent_aggregation"] = "deepsets"
    elif method == "fmca_av_matched_head":
        model_override["parent_aggregation"] = "mean"
        base_hidden = 2048 if dataset == "imagenet100" else 512
        model_override["f_head_hidden_dims"] = [base_hidden * (int(action["views"]) + 1) // 2]
    elif method == "hfmca_style": model_override["parent_aggregation"] = "concat"
    elif method == "regular_fmca":
        model_override["parent_aggregation"] = "raw"
        override["data"] = {"num_views": int(action["views"]), "include_raw_parent": True}
    if action.get("backbone"): model_override["backbone"] = str(action["backbone"])
    if action.get("aggregation"): model_override["parent_aggregation"] = str(action["aggregation"])
    if model_override: override["model"] = model_override
    if dataset == "imagenet100":
        override["data"] = {
            "num_views": int(action["views"]), "batch_size": 64,
            "include_raw_parent": method == "regular_fmca",
        }
        override["trainer"] = {"max_epochs": int(action["target"]), "devices": 2, "strategy": "ddp",
                               "limit_train_batches": 1.0, "limit_val_batches": 1.0, "max_steps": -1,
                               "checkpoint_save_top_k": 0 if int(action["target"]) < int(spec["epochs"]) else 1}
    return override


def evaluation_override(action: dict[str, object]) -> dict[str, object]:
    dataset = str(action["dataset"]); method = str(action["method"])
    override: dict[str, object] = {
        "experiment": {"name": f"{dataset}-{method}-v{action['views']}-confirmatory"},
        "data": {"num_views": int(action["views"])},
        "probe": {"max_epochs": 100, "devices": 1, "accelerator": "gpu",
                  "limit_train_batches": 1.0, "limit_val_batches": 1.0, "limit_test_batches": 1.0},
    }
    if method not in FMCA_METHODS: override["experiment"] = {"name": f"{dataset}-{method}-v{action['views']}-confirmatory", "method": method}
    model_override: dict[str, object] = {}
    if method == "fmca_av_deepsets":
        model_override["parent_aggregation"] = "deepsets"
    elif method == "fmca_av_matched_head":
        model_override["parent_aggregation"] = "mean"
        base_hidden = 2048 if dataset == "imagenet100" else 512
        model_override["f_head_hidden_dims"] = [base_hidden * (int(action["views"]) + 1) // 2]
    elif method == "hfmca_style": model_override["parent_aggregation"] = "concat"
    elif method == "regular_fmca": model_override["parent_aggregation"] = "raw"
    if action.get("backbone"): model_override["backbone"] = str(action["backbone"])
    if action.get("aggregation"): model_override["parent_aggregation"] = str(action["aggregation"])
    if model_override: override["model"] = model_override
    return override


def command_for(action: dict[str, object], state: dict[str, object]) -> tuple[str, int, str, list[str]]:
    dataset = str(action["dataset"]); method = str(action["method"]); kind = str(action["kind"]); spec = DATASETS[dataset]
    config = str(spec["config"]); gpus = int(spec["gpus"]); seed = int(action["seed"]); key = str(action["key"])
    checkpoint = str(dict(state["last_checkpoints"]).get(key, ""))
    profile = "imagenet_ddp" if dataset == "imagenet100" and kind == "train" else ("imagenet" if dataset == "imagenet100" else "default")
    architecture = ""
    if action.get("backbone"): architecture = f"-{action['backbone']}-{action.get('aggregation', 'mean')}"
    stem = f"formal-{dataset}-{method}{architecture}-v{action['views']}-seed{action['seed_index']}"
    if kind == "train":
        override_json = json.dumps(training_override(action), separators=(",", ":")); target = int(action["target"])
        if method in FMCA_METHODS:
            payload = ["-m", "scripts.run_fmca_pipeline", "--config", config, "--seed", str(seed), "--overrides-json", override_json]
            if checkpoint: payload += ["--resume", checkpoint]
            if target < int(spec["epochs"]): payload.append("--train-only")
            command = ([TORCHRUN, "--standalone", "--nnodes=1", f"--nproc_per_node={gpus}", *payload]
                       if gpus > 1 else [PYTHON, *payload])
        else:
            payload = ["-m", "fmca_av.baseline_cli", "train", "--config", config, "--seed", str(seed),
                       "--overrides-json", override_json]
            if checkpoint: payload += ["--resume", checkpoint]
            command = ([TORCHRUN, "--standalone", "--nnodes=1", f"--nproc_per_node={gpus}", *payload]
                       if gpus > 1 else [PYTHON, *payload])
        return f"{stem}-epoch-{target}", gpus, profile, command
    override_json = json.dumps(evaluation_override(action), separators=(",", ":"))
    if kind == "probe":
        if method in FMCA_METHODS:
            command = ["env", "FMCA_CONFIG_OVERRIDES=" + override_json, f"FMCA_SEED_OVERRIDE={seed}", PYTHON, "-m",
                       "fmca_av.cli", "linear-probe", "--config", config, "--checkpoint", checkpoint]
        else:
            command = [PYTHON, "-m", "fmca_av.baseline_cli", "linear-probe", "--config", config,
                       "--checkpoint", checkpoint, "--seed", str(seed), "--overrides-json", override_json]
        return f"{stem}-linear-probe", 1, profile, command
    command = ["env", "FMCA_CONFIG_OVERRIDES=" + override_json, f"FMCA_SEED_OVERRIDE={seed}", PYTHON, "-m",
               "fmca_av.cli", "knn", "--config", config, "--checkpoint", checkpoint, "--workers", "8",
               "--batch-size", "256", "--bank-chunk-size", "8192"]
    return f"{stem}-knn", 1, profile, command


def submit_action(action: dict[str, object], state: dict[str, object], retry_from: str = "") -> str:
    name, gpus, profile, command = command_for(action, state)
    profile_args = ["--profile", profile] if profile != "default" else []
    retry_args = ["--retry-from", retry_from] if retry_from else []
    return submit(["python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", str(gpus),
                   *profile_args, *retry_args, "--", *command])


def submit_successor(state_file: Path, dependencies: list[str], index: int) -> str:
    return submit(["python3", "-m", "harness.cli", "watch", "--name", f"formal-ssl-chain-step-{index:04d}", "--",
                   PYTHON, "-m", "scripts.formal_ssl_state_machine", "--state-file", str(state_file),
                   "--dependencies", ",".join(dependencies)])


def action_gpus(action: dict[str, object]) -> int:
    return int(DATASETS[str(action["dataset"])]["gpus"]) if str(action["kind"]) == "train" else 1


def formal_gpu_capacity(config: dict[str, object]) -> int:
    """Reserve part of the global budget for independent experiment chains."""
    global_capacity = int(config["max_gpus"])
    requested = int(config.get("formal_ssl_max_gpus", min(4, global_capacity)))
    if requested < 1:
        raise ValueError("formal_ssl_max_gpus must be positive")
    return min(global_capacity, requested)


def gpu_capacity() -> int:
    config = json.loads(Path("harness/config.json").read_text(encoding="utf-8"))
    return formal_gpu_capacity(config)


def take_batch(
    state: dict[str, object],
    plan: list[dict[str, object]],
    available_gpus: int = 6,
    active_keys: set[str] | None = None,
) -> list[dict[str, object]]:
    """Select independent actions that fit beside the currently active actions."""
    retries = list(state.get("retry_queue", []))
    source = ([{**record, "origin": "retry"} for record in retries] if retries else [
        {"action": action, "attempt": 1, "infrastructure_attempt": 0, "origin": "plan"}
        for action in plan[int(state["action_index"]):]
    ])
    selected: list[dict[str, object]] = []
    used = 0
    keys: set[str] = set(active_keys or set())
    for record in source:
        action = dict(record["action"]); gpus = action_gpus(action); key = str(action["key"])
        if used + gpus > available_gpus or key in keys: break
        selected.append({
            "action": action, "attempt": int(record["attempt"]),
            "infrastructure_attempt": int(record.get("infrastructure_attempt", 0)),
            "retry_from": str(record.get("retry_from", "")),
            "origin": record["origin"],
        })
        used += gpus; keys.add(key)
        if used == available_gpus: break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--state-file", default="results/orchestration/formal_ssl_state.json")
    parser.add_argument("--dependencies", default=",".join(DEFAULT_DEPENDENCIES)); args = parser.parse_args()
    dependencies = [value for value in args.dependencies.split(",") if value]; state_file = Path(args.state_file).resolve()
    state = load_state(state_file, dependencies); chain = list(state["chain_runs"]); current_chain_run = os.environ["FMCA_HARNESS_RUN_ID"]
    if current_chain_run not in chain: chain.append(current_chain_run)
    state["chain_runs"] = chain; save_state(state_file, state)
    if not bool(state["dependencies_complete"]): wait_dependencies(list(state["dependencies"])); state["dependencies_complete"] = True; save_state(state_file, state)
    current = list(state.get("current_runs", []))
    if current:
        if bool(state.get("pause_requested")):
            # A priority change must not kill scientific work already inside
            # Slurm.  Drain exactly the recorded in-flight batch, reconcile its
            # checkpoints, and stop before selecting any successor action.
            observed = wait_terminal([str(record["run_id"]) for record in current])
        else:
            observed = {str(record["run_id"]): run_state(str(record["run_id"])) for record in current}
        active_gpus_before_refresh = sum(
            action_gpus(dict(record["action"]))
            for record in current
            if observed[str(record["run_id"])] not in {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"}
        )
        if (not any(value in {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"} for value in observed.values())
                and active_gpus_before_refresh >= gpu_capacity()):
            observed = wait_for_progress([str(record["run_id"]) for record in current])
        checkpoints = dict(state["last_checkpoints"]); completed = list(state["completed"])
        retry_queue = list(state.get("retry_queue", []))
        remaining = []
        for record in current:
            run_id = str(record["run_id"]); action = dict(record["action"]); attempt = int(record["attempt"])
            infrastructure_attempt = int(record.get("infrastructure_attempt", 0))
            terminal = observed[run_id]
            if terminal not in {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"}:
                remaining.append(record)
                continue
            if terminal == "SUCCEEDED":
                if str(action["kind"]) == "train":
                    candidate = last_checkpoint(run_id)
                    if not candidate: raise RuntimeError(f"successful training action {run_id} has no checkpoint")
                    checkpoints[str(action["key"])] = candidate
                completed.append({"action": action, "run_id": run_id, "state": terminal})
            else:
                candidate = last_checkpoint(run_id) if str(action["kind"]) == "train" else ""
                if candidate: checkpoints[str(action["key"])] = candidate
                retry = retry_record(action, run_id, terminal, attempt, infrastructure_attempt)
                if retry is None:
                    kind = "infrastructure" if is_infrastructure_failure(run_id, terminal) else "scientific"
                    state["state"] = "FAILED"; state["failure"] = {
                        "action": action, "run_id": run_id, "terminal": terminal,
                        "failure_kind": kind, "scientific_attempt": attempt,
                        "infrastructure_attempt": infrastructure_attempt,
                    }
                    save_state(state_file, state); raise RuntimeError(f"formal SSL action exhausted {kind} retries: {run_id}")
                retry_queue.append(retry)
        state["last_checkpoints"] = checkpoints; state["completed"] = completed
        state["retry_queue"] = retry_queue; state["current_runs"] = remaining; save_state(state_file, state)
    if bool(state.get("pause_requested")):
        state["state"] = "PAUSED"
        state["pause_reason"] = str(state.get("pause_reason", "operator priority change"))
        state["successor_run"] = ""
        save_state(state_file, state)
        return 0
    plan = actions()
    if int(state["action_index"]) >= len(plan) and not state.get("retry_queue"):
        state["state"] = "SUCCEEDED"; save_state(state_file, state); return 0
    current = list(state.get("current_runs", []))
    active_gpus = sum(action_gpus(dict(record["action"])) for record in current)
    active_keys = {str(dict(record["action"])["key"]) for record in current}
    batch = take_batch(state, plan, available_gpus=max(0, gpu_capacity() - active_gpus), active_keys=active_keys)
    if not batch:
        successor = submit_successor(state_file, dependencies, int(state["action_index"]))
        state["successor_run"] = successor; save_state(state_file, state); return 0
    launched = list(current)
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
        state["current_runs"] = launched; save_state(state_file, state)
    index = int(state["action_index"])
    successor = submit_successor(state_file, dependencies, index); state["successor_run"] = successor; save_state(state_file, state); return 0


if __name__ == "__main__": raise SystemExit(main())
