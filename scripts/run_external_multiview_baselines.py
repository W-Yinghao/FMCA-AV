#!/usr/bin/env python3
"""Bounded CIFAR-10 FastSSL/FroSSL experiment controller.

This lightweight login-node controller only submits through the existing Slurm
harness and refreshes state every 300 seconds.  It never runs model code itself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
SEEDS = (20260821, 20260822, 20260823)
METHODS = {
    "fastssl_barlow_twins": "configs/ssl/cifar10_fastssl_barlow_twins.json",
    "fastssl_vicreg": "configs/ssl/cifar10_fastssl_vicreg.json",
    "frossl": "configs/ssl/cifar10_frossl.json",
}
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def method_override(method: str, views: int, seed: int, smoke: bool = False) -> dict[str, object]:
    override: dict[str, object] = {
        "experiment": {"name": f"cifar10-{method}-v{views}", "method": method},
        "seed": seed,
        "data": {"num_views": views},
    }
    if method == "frossl":
        override["objective"] = {"invariance_weight": 1.4 if views == 2 else 2.0}
    if smoke:
        override["data"] = {"num_views": views, "batch_size": 8, "num_workers": 2}
        override["trainer"] = {
            "max_epochs": 1,
            "limit_train_batches": 2,
            "limit_val_batches": 1,
            "checkpoint_save_top_k": 0,
        }
        override["optimizer"] = {"scheduler": "none"}
    return override


def actions() -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    smoke_keys = []
    for method, config in METHODS.items():
        key = f"smoke:{method}"
        smoke_keys.append(key)
        values.append({"key": key, "kind": "smoke", "method": method, "config": config, "views": 8, "seed": SEEDS[0], "dependencies": []})
    for method, config in METHODS.items():
        for views in (2, 8):
            values.append({
                "key": f"flops:{method}:v{views}", "kind": "flops", "method": method,
                "config": config, "views": views, "dependencies": smoke_keys,
            })
        for views in (2, 8):
            for seed_index, seed in enumerate(SEEDS, 1):
                train_key = f"train:{method}:v{views}:s{seed_index}"
                values.append({
                    "key": train_key, "kind": "train", "method": method, "config": config,
                    "views": views, "seed": seed, "seed_index": seed_index, "dependencies": smoke_keys,
                })
                for kind in ("probe", "knn", "diagnostics"):
                    values.append({
                        "key": f"{kind}:{method}:v{views}:s{seed_index}", "kind": kind,
                        "method": method, "config": config, "views": views, "seed": seed,
                        "seed_index": seed_index, "train_key": train_key, "dependencies": [train_key],
                    })
    final_dependencies = [str(value["key"]) for value in values if value["kind"] not in {"smoke", "train"}]
    values.append({"key": "aggregate", "kind": "aggregate", "dependencies": final_dependencies})
    return values


def refresh() -> None:
    subprocess.run(["python3", "-m", "harness.cli", "status"], check=True, stdout=subprocess.DEVNULL)


def run_state(run_id: str) -> str:
    path = Path("runs") / run_id / "status.json"
    return str(read(path)["state"]) if path.is_file() else "BLOCKED"


def checkpoint(record: dict[str, object], records: dict[str, dict[str, object]]) -> str:
    train = records[str(record["train_key"])]
    result_path = Path("runs") / str(train["run_id"]) / "artifacts" / "train_result.json"
    payload = read(result_path)
    if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing non-current checkpoint from {train['run_id']}")
    value = payload.get("last_checkpoint") or payload.get("best_checkpoint")
    if not value or not Path(str(value)).is_file():
        raise RuntimeError(f"missing checkpoint from {train['run_id']}")
    return str(value)


def command(record: dict[str, object], records: dict[str, dict[str, object]], state_file: Path) -> tuple[str, int, list[str]]:
    kind = str(record["kind"])
    if kind == "aggregate":
        return "external-baselines-aggregate", 0, [
            PYTHON, "-m", "scripts.render_external_multiview_baselines", "--state-file", str(state_file),
        ]
    method = str(record["method"]); views = int(record["views"]); config = str(record["config"])
    stem = f"external-c10-{method}-v{views}"
    if kind == "smoke":
        override = method_override(method, views, int(record["seed"]), smoke=True)
        return f"{stem}-smoke", 1, [
            PYTHON, "-m", "fmca_av.baseline_cli", "train", "--config", config,
            "--seed", str(record["seed"]), "--overrides-json", json.dumps(override, separators=(",", ":")),
        ]
    if kind == "flops":
        return f"{stem}-flops", 1, [
            PYTHON, "-m", "scripts.profile_external_baseline_flops", "--config", config,
            "--views", str(views), "--batch", "2",
        ]
    seed = int(record["seed"]); seed_index = int(record["seed_index"])
    override = method_override(method, views, seed)
    override_json = json.dumps(override, separators=(",", ":"))
    if kind == "train":
        return f"{stem}-seed{seed_index}-pretrain", 1, [
            PYTHON, "-m", "fmca_av.baseline_cli", "train", "--config", config,
            "--seed", str(seed), "--overrides-json", override_json,
        ]
    source = checkpoint(record, records)
    if kind == "probe":
        return f"{stem}-seed{seed_index}-linear-probe", 1, [
            PYTHON, "-m", "fmca_av.baseline_cli", "linear-probe", "--config", config,
            "--checkpoint", source, "--seed", str(seed), "--overrides-json", override_json,
        ]
    if kind == "knn":
        return f"{stem}-seed{seed_index}-knn", 1, [
            "env", "FMCA_CONFIG_OVERRIDES=" + override_json, f"FMCA_SEED_OVERRIDE={seed}", PYTHON,
            "-m", "fmca_av.cli", "knn", "--config", config, "--checkpoint", source,
            "--workers", "8", "--batch-size", "256", "--bank-chunk-size", "8192",
        ]
    if kind == "diagnostics":
        return f"{stem}-seed{seed_index}-diagnostics", 1, [
            "env", "FMCA_CONFIG_OVERRIDES=" + override_json, f"FMCA_SEED_OVERRIDE={seed}", PYTHON,
            "-m", "scripts.evaluate_baseline_diagnostics", "--config", config, "--checkpoint", source,
        ]
    raise ValueError(f"unknown action kind {kind}")


def submit(record: dict[str, object], records: dict[str, dict[str, object]], state_file: Path) -> str:
    name, gpus, payload = command(record, records, state_file)
    argv = ["python3", "-m", "harness.cli", "submit", "--name", name, "--gpus", str(gpus)]
    if gpus:
        # This profile is a resource policy: it permits A100/L40S/H100 and excludes V100.
        argv.extend(["--profile", "imagenet"])
    argv.extend(["--", *payload])
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        message = result.stderr.strip()
        if "GPU limit exceeded" in message or "Slurm job limit reached" in message:
            return ""
        raise RuntimeError(message or result.stdout.strip())
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", required=True)
    args = parser.parse_args()
    state_file = Path(args.state_file)
    state = read(state_file) if state_file.is_file() else {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "scope": "CIFAR-10 FastSSL-Barlow-Twins, FastSSL-VICReg, FroSSL",
        "poll_seconds": POLL_SECONDS,
        "state": "RUNNING",
        "created_at": now(),
        "records": {},
    }
    if state.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError("refusing a controller state from another correctness version")
    records = dict(state.get("records", {}))
    definitions = {str(value["key"]): value for value in actions()}
    for key, definition in definitions.items():
        records.setdefault(key, {**definition, "state": "WAITING"})
    state["records"] = records; state["state"] = "RUNNING"; write(state_file, state)

    while True:
        # A brand-new state has nothing to poll: submit its root smokes first.
        # Every later cycle has run IDs and therefore performs one squeue-backed
        # harness refresh after the fixed 300-second sleep.
        if any(record.get("run_id") for record in records.values()):
            refresh()
            state["last_polled_at"] = now()
        changed = False
        for record in records.values():
            run_id = str(record.get("run_id", ""))
            if run_id and record.get("state") not in TERMINAL:
                current = run_state(run_id)
                if current != record.get("state"):
                    record["state"] = current; record["updated_at"] = now(); changed = True
        failures = [record for record in records.values() if record.get("state") in {"FAILED", "STOPPED", "BLOCKED"}]
        if failures:
            state["state"] = "FAILED"; state["failed_at"] = now(); write(state_file, state)
            raise RuntimeError("external baseline action failed: " + json.dumps(failures, sort_keys=True))
        if records["aggregate"].get("state") == "SUCCEEDED":
            state["state"] = "SUCCEEDED"; state["completed_at"] = now(); write(state_file, state)
            return 0
        if changed:
            write(state_file, state)

        submitted = False
        for key in definitions:
            record = records[key]
            if record.get("state") != "WAITING":
                continue
            dependencies = [records[str(item)].get("state") for item in record.get("dependencies", [])]
            if not all(value == "SUCCEEDED" for value in dependencies):
                continue
            run_id = submit(record, records, state_file)
            if not run_id:
                break
            record["run_id"] = run_id; record["state"] = "QUEUED"; record["submitted_at"] = now()
            write(state_file, state); submitted = True
        # Status is sampled only once per cycle, with the project-wide fixed interval.
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
