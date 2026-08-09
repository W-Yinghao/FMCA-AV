#!/usr/bin/env python3
"""Run the bounded three-seed CIFAR-10 strong-baseline priority screen."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


POLL_SECONDS = 300
PYTHON = "/projects/EEG-foundation-model/yinghao/FMCA-AV/envs/lightning/bin/python"
CONFIG = "configs/ssl/cifar10_reference.json"
METHODS = ("simclr", "vicreg", "dino", "byol")
SEEDS = (20280001, 20280002, 20280003)
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "BLOCKED"}


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


def wait_all(run_ids: list[str], label: str) -> None:
    while True:
        time.sleep(POLL_SECONDS); refresh()
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
        time.sleep(POLL_SECONDS); refresh()


def checkpoint(run_id: str) -> tuple[str, dict[str, object]]:
    result = read(Path("runs") / run_id / "artifacts" / "train_result.json")
    if result.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing pre-fix baseline source {run_id}")
    value = result.get("last_checkpoint") or result.get("best_checkpoint")
    if not value or not Path(str(value)).is_file():
        raise RuntimeError(f"missing baseline checkpoint {run_id}")
    return str(value), result


def training_command(method: str, seed_index: int) -> list[str]:
    override = {
        "experiment": {"name": f"cifar10-{method}-v8-priority", "method": method},
        "data": {"num_views": 8},
        "trainer": {"max_epochs": 200, "checkpoint_save_top_k": 1},
        "optimizer": {"scheduler_t_max": 800},
    }
    return [PYTHON, "-m", "fmca_av.baseline_cli", "train", "--config", CONFIG,
            "--seed", str(SEEDS[seed_index - 1]), "--overrides-json", json.dumps(override, separators=(",", ":"))]


def probe_command(method: str, seed_index: int, source: str) -> list[str]:
    override = {
        "experiment": {"name": f"cifar10-{method}-v8-priority", "method": method},
        "data": {"num_views": 8},
        "probe": {"max_epochs": 100, "devices": 1, "accelerator": "gpu",
                  "limit_train_batches": 1.0, "limit_val_batches": 1.0, "limit_test_batches": 1.0},
    }
    return [PYTHON, "-m", "fmca_av.baseline_cli", "linear-probe", "--config", CONFIG,
            "--checkpoint", source, "--seed", str(SEEDS[seed_index - 1]),
            "--overrides-json", json.dumps(override, separators=(",", ":"))]


def select_methods(rows: list[dict[str, object]], margin: float = 0.01, maximum: int = 2) -> list[str]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    summaries = []
    for method, values in grouped.items():
        if len(values) != 3:
            raise RuntimeError(f"baseline {method} lacks three paired seeds")
        summaries.append((
            method,
            statistics.fmean(float(value["test_accuracy"]) for value in values),
            statistics.fmean(float(value["gpu_hours"]) for value in values),
        ))
    best = max(value[1] for value in summaries)
    eligible = [value for value in summaries if value[1] >= best - margin]
    eligible.sort(key=lambda value: (-value[1], value[2], value[0]))
    return [value[0] for value in eligible[:maximum]]


def render(rows: list[dict[str, object]], selected: list[str]) -> None:
    output = Path(f"results/postfix/{SCIENTIFIC_CORRECTNESS_VERSION}/e5")
    output.mkdir(parents=True, exist_ok=True)
    fields = ["method", "views", "seed_index", "seed", "train_run", "probe_run",
              "test_accuracy", "validation_accuracy", "gpu_hours", "encoded_views",
              "trainable_parameters", "selected"]
    path = output / "priority_baseline_200epoch_gate.csv"; temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({**row, "selected": str(row["method"] in selected).lower()})
    temporary.replace(path)
    write(output / "priority_baseline_200epoch_gate.json", {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "paired_seed_indices": [1, 2, 3], "selection_margin": 0.01,
        "maximum_selected": 2, "selected_methods": selected, "rows": rows,
    })


def parse_initial(value: str) -> dict[str, str]:
    result = {}
    for item in value.split(","):
        if not item:
            continue
        key, run_id = item.split("=", 1); result[key] = run_id
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--state-file", required=True)
    parser.add_argument("--initial-runs", default=""); args = parser.parse_args()
    path = Path(args.state_file)
    state = read(path) if path.is_file() else {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "state": "RUNNING", "chain_runs": [], "records": [],
    }
    if state.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing legacy baseline priority state {path}")
    chain = list(state.get("chain_runs", [])); current = os.environ["FMCA_HARNESS_RUN_ID"]
    if current not in chain:
        chain.append(current)
    state["chain_runs"] = chain; state["state"] = "RUNNING"
    records = list(state.get("records", [])); existing = {(str(r["method"]), int(r["seed_index"])) for r in records}
    initial = parse_initial(args.initial_runs)
    for method in METHODS:
        for seed_index in range(1, 4):
            if (method, seed_index) in existing:
                continue
            key = f"{method}:{seed_index}"
            run_id = initial.get(key) or submit(
                f"priority-e5-cifar10-{method}-v8-seed{seed_index}-epoch200",
                training_command(method, seed_index),
            )
            records.append({"method": method, "views": 8, "seed_index": seed_index,
                            "seed": SEEDS[seed_index - 1], "train_run": run_id})
            existing.add((method, seed_index)); state["records"] = records; write(path, state)
    wait_all([str(record["train_run"]) for record in records], "priority baseline training")
    for record in records:
        if record.get("probe_run"):
            continue
        source, _ = checkpoint(str(record["train_run"]))
        record["probe_run"] = submit(
            f"priority-e5-cifar10-{record['method']}-v8-seed{record['seed_index']}-probe-epoch200",
            probe_command(str(record["method"]), int(record["seed_index"]), source),
        )
        state["records"] = records; write(path, state)
    wait_all([str(record["probe_run"]) for record in records], "priority baseline probes")
    rows = []
    for record in records:
        _, train = checkpoint(str(record["train_run"]))
        probe = read(Path("runs") / str(record["probe_run"]) / "artifacts" / "probe_result.json")
        rows.append({
            **record, "test_accuracy": probe["test_accuracy"],
            "validation_accuracy": probe["best_validation_accuracy"],
            "gpu_hours": train["gpu_hours"], "encoded_views": train["encoded_views"],
            "trainable_parameters": train["trainable_parameters"],
        })
    selected = select_methods(rows); render(rows, selected)
    state["selected_methods"] = selected; state["state"] = "SUCCEEDED"; write(path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
