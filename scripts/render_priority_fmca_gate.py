#!/usr/bin/env python3
"""Render the preregistered 200-epoch FMCA M=2/M=8 selection gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def choose(summary: list[dict[str, object]], margin: float = 0.01, maximum: int = 2) -> list[str]:
    best = max(float(row["test_accuracy_mean"]) for row in summary)
    eligible = [row for row in summary if float(row["test_accuracy_mean"]) >= best - margin]
    eligible.sort(key=lambda row: (
        -float(row["test_accuracy_mean"]), float(row["gpu_hours_mean"]),
        int(row["encoded_views_mean"]), str(row["configuration"]),
    ))
    return [str(row["configuration"]) for row in eligible[:maximum]]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--state-file", required=True); args = parser.parse_args()
    state = read(Path(args.state_file))
    if state.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION or state.get("state") != "SUCCEEDED":
        raise RuntimeError("priority checkpoint probe state is incomplete or from the wrong scientific version")
    rows = []
    for record in list(state["probes"]):
        action = dict(record["action"]); train_run = str(record["source_run"]); probe_run = str(record["run_id"])
        train = read(Path("runs") / train_run / "artifacts" / "train_result.json")
        probe = read(Path("runs") / probe_run / "artifacts" / "probe_result.json")
        if train.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            raise RuntimeError(f"wrong-version train result {train_run}")
        if probe.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            raise RuntimeError(f"wrong-version probe result {probe_run}")
        rows.append({
            "configuration": f"fmca_av_m{action['views']}", "views": action["views"],
            "seed_index": action["seed_index"], "seed": action["seed"],
            "train_run": train_run, "probe_run": probe_run,
            "test_accuracy": probe["test_accuracy"],
            "validation_accuracy": probe["best_validation_accuracy"],
            "gpu_hours": train["gpu_hours"], "encoded_views": train["encoded_views"],
            "trainable_parameters": train["trainable_parameters"],
        })
    groups = {}
    for row in rows:
        groups.setdefault(str(row["configuration"]), []).append(row)
    summary = []
    for configuration, values in sorted(groups.items()):
        if len(values) != 3:
            raise RuntimeError(f"{configuration} lacks three paired seeds")
        summary.append({
            "configuration": configuration, "views": values[0]["views"], "paired_seeds": 3,
            "test_accuracy_mean": statistics.fmean(float(value["test_accuracy"]) for value in values),
            "test_accuracy_std": statistics.stdev(float(value["test_accuracy"]) for value in values),
            "gpu_hours_mean": statistics.fmean(float(value["gpu_hours"]) for value in values),
            "encoded_views_mean": round(statistics.fmean(float(value["encoded_views"]) for value in values)),
            "trainable_parameters": values[0]["trainable_parameters"],
        })
    selected = choose(summary)
    output = Path(f"results/postfix/{SCIENTIFIC_CORRECTNESS_VERSION}/e5"); output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "priority_fmca_200epoch_gate.csv"; temporary = csv_path.with_suffix(".csv.tmp")
    fields = ["configuration", "views", "seed_index", "seed", "train_run", "probe_run",
              "test_accuracy", "validation_accuracy", "gpu_hours", "encoded_views", "trainable_parameters"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(csv_path)
    write(output / "priority_fmca_200epoch_gate.json", {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "checkpoint_epoch": 200, "paired_seed_indices": [1, 2, 3],
        "selection_margin": 0.01, "maximum_selected": 2,
        "selected_configurations": selected, "summary": summary,
    })
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "priority_fmca_gate", "selected": selected}) + "\n")
    print(json.dumps({"selected": selected, "summary": summary}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
