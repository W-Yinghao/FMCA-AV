#!/usr/bin/env python3
"""Render CIFAR numerical/objective ablation assets from completed E3 runs."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


SOURCES = (
    Path("runs/20260807-062417_launch-e3-cifar-numerics-wave/artifacts/submitted.json"),
    Path("runs/20260807-073411_recover-interrupted-e3-tsd-coco-v2/artifacts/e3_submitted.json"),
)
RESULTS_ROOT = Path(os.environ.get(
    "FMCA_RESULTS_ROOT", f"results/postfix/{SCIENTIFIC_CORRECTNESS_VERSION}",
))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8"); temporary.replace(path)


def spectral_metrics(values: list[float]) -> dict[str, float | int]:
    positive = [max(0.0, float(value)) for value in values]
    numerical = [value for value in positive if value > 1e-6]
    total = sum(positive); probabilities = [value / total for value in positive if value > 0 and total > 0]
    entropy = -sum(value * math.log(value) for value in probabilities)
    return {
        "trace": total, "numerical_rank": len(numerical), "effective_rank": math.exp(entropy) if probabilities else 0.0,
        "condition_number": max(numerical) / min(numerical) if numerical else float("inf"),
        "minimum_eigenvalue": min(positive) if positive else 0.0, "maximum_eigenvalue": max(positive) if positive else 0.0,
    }


def main() -> int:
    by_tag: dict[str, dict[str, object]] = {}
    for source in SOURCES:
        if not source.is_file(): continue
        for record in json.loads(source.read_text(encoding="utf-8")):
            by_tag[str(record["tag"])] = {**record, "dataset": "cifar10"}
    imagenet_state = Path("results/orchestration/e3_imagenet100_recheck_state.json")
    if imagenet_state.is_file():
        for record in json.loads(imagenet_state.read_text(encoding="utf-8")).get("submitted", []):
            tag = "imagenet100-" + str(record["key"])
            by_tag[tag] = {
                "tag": tag, "dataset": "imagenet100", "seed": record["seed"],
                "run_id": record["run_id"], "override": record["override"],
            }
    rows = []
    for tag, record in sorted(by_tag.items()):
        run_id = str(record["run_id"]); run_dir = Path("runs") / run_id
        train_path = run_dir / "artifacts" / "train_result.json"
        if not train_path.is_file() or read_json(train_path).get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            continue
        status = read_json(run_dir / "status.json")
        row: dict[str, object] = {"dataset": record.get("dataset", "cifar10"), "tag": tag, "seed": record["seed"], "run_id": run_id, "state": status["state"],
                                  "override_json": json.dumps(record["override"], sort_keys=True, separators=(",", ":")),
                                  "duration_seconds": "", "best_validation_score": "", "trace": "", "numerical_rank": "",
                                  "effective_rank": "", "condition_number": "", "minimum_eigenvalue": "", "maximum_eigenvalue": ""}
        if status.get("start_time") and status.get("end_time"):
            from datetime import datetime
            row["duration_seconds"] = (datetime.fromisoformat(status["end_time"]) - datetime.fromisoformat(status["start_time"])).total_seconds()
        evaluation_path = run_dir / "artifacts" / "evaluation.json"
        if status["state"] == "SUCCEEDED" and train_path.is_file():
            row["best_validation_score"] = read_json(train_path).get("best_validation_score", "")
        if status["state"] == "SUCCEEDED" and evaluation_path.is_file():
            evaluation = read_json(evaluation_path)
            values = evaluation.get("test_empirical_eigenvalues", []) if evaluation.get("scientific_correctness_version") == SCIENTIFIC_CORRECTNESS_VERSION else []
            if values:
                row.update(spectral_metrics(values))
        rows.append(row)
    output = RESULTS_ROOT / "e3"; output.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "tag", "seed", "run_id", "state", "override_json", "duration_seconds", "best_validation_score", "trace",
              "numerical_rank", "effective_rank", "condition_number", "minimum_eigenvalue", "maximum_eigenvalue"]
    temporary = output / "cifar_numerics_table.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    temporary.replace(output / "cifar_numerics_table.csv")
    estimator_rows = []
    for control_path in sorted(Path("runs").glob("*/artifacts/e3_estimator_controls.json")):
        status_path = control_path.parents[1] / "status.json"
        if not status_path.is_file() or json.loads(status_path.read_text(encoding="utf-8")).get("state") != "SUCCEEDED":
            continue
        run_id = control_path.parents[1].name
        payload = read_json(control_path)
        if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            continue
        for record in payload.get("records", []):
            estimator_rows.append({"run_id": run_id, **dict(record)})
    estimator_fields = sorted({key for row in estimator_rows for key in row}) if estimator_rows else ["run_id"]
    temporary = output / "estimator_controls.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=estimator_fields, lineterminator="\n"); writer.writeheader(); writer.writerows(estimator_rows)
    temporary.replace(output / "estimator_controls.csv")
    grouped_controls: dict[tuple[object, ...], list[dict[str, object]]] = {}
    control_keys = ("dataset", "samples", "dimension", "centered", "precision", "whitening", "ridge_rule", "objective")
    for record in estimator_rows:
        grouped_controls.setdefault(tuple(record.get(key, "") for key in control_keys), []).append(record)
    control_summaries = []
    for key, records in sorted(grouped_controls.items(), key=lambda item: str(item[0])):
        successful_records = [record for record in records if record.get("state") == "SUCCEEDED"]
        summary: dict[str, object] = dict(zip(control_keys, key))
        summary.update({"runs": len(records), "successful_runs": len(successful_records), "failure_rate": 1.0 - len(successful_records) / len(records)})
        for metric in ("spectrum_mae", "spectrum_relative_error", "score", "rf_condition", "rg_condition", "minimum_variance"):
            values = [float(record[metric]) for record in successful_records if record.get(metric, "") != ""]
            mean = statistics.fmean(values) if values else ""
            std = statistics.stdev(values) if len(values) > 1 else 0.0 if values else ""
            summary[metric + "_mean"] = mean; summary[metric + "_std"] = std
            summary[metric + "_ci95_half_width"] = 1.96 * float(std) / math.sqrt(len(values)) if values else ""
        control_summaries.append(summary)
    control_summary_fields = sorted({key for row in control_summaries for key in row}) if control_summaries else ["dataset"]
    temporary = output / "estimator_controls_summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=control_summary_fields, lineterminator="\n"); writer.writeheader(); writer.writerows(control_summaries)
    temporary.replace(output / "estimator_controls_summary.csv")
    numerical_rows = []
    constant_mode_controls = []
    for numerical_path in sorted(Path("runs").glob("*/artifacts/e3_numerics.json")):
        status_path = numerical_path.parents[1] / "status.json"
        if not status_path.is_file() or read_json(status_path).get("state") != "SUCCEEDED":
            continue
        payload = read_json(numerical_path)
        if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            continue
        run_id = numerical_path.parents[1].name
        numerical_rows.extend({"run_id": run_id, **dict(record)} for record in payload.get("records", []))
        constant_mode_controls.append({"run_id": run_id, **dict(payload.get("constant_mode_failure_control", {}))})
    numerical_fields = sorted({key for row in numerical_rows for key in row}) if numerical_rows else ["run_id"]
    temporary = output / "numerical_ablation.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=numerical_fields, lineterminator="\n"); writer.writeheader(); writer.writerows(numerical_rows)
    temporary.replace(output / "numerical_ablation.csv")
    atomic_text(output / "constant_mode_controls.json", json.dumps(constant_mode_controls, indent=2, sort_keys=True) + "\n")
    successful = [row for row in rows if row["state"] == "SUCCEEDED" and row["best_validation_score"] != ""]
    width = 1200; height = max(360, 80 + 24 * len(successful)); left = 360; chart = 760
    values = [float(row["best_validation_score"]) for row in successful]; maximum = max(values, default=1.0)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<style>.label{font:11px sans-serif}.title{font:600 16px sans-serif}.bar{fill:#2855a6}</style>',
           '<text x="20" y="28" class="title">E3 CIFAR-10 and ImageNet-100 numerical ablations</text>']
    for index, row in enumerate(successful):
        y = 55 + 24 * index; value = float(row["best_validation_score"]); length = chart * value / max(maximum, 1e-12)
        svg.append(f'<text x="20" y="{y+12}" class="label">{row["tag"]}</text><rect x="{left}" y="{y}" width="{length:.2f}" height="15" class="bar"/><text x="{left+length+5:.2f}" y="{y+12}" class="label">{value:.3f}</text>')
    svg.append("</svg>"); atomic_text(output / "cifar_numerics.svg", "".join(svg))
    atomic_text(output / "cifar_numerics_caption.txt",
                "Post-fix E3 numerical and objective ablations. Only artifacts carrying the required scientific-correctness version are included. The estimator-control tables report Gaussian/finite-channel centered, whitening, adaptive-ridge, objective, precision, batch/sample and post-hoc spectral controls with failure rates and 95% confidence intervals; numerical_ablation.csv and constant_mode_controls.json retain the exact Hermite-feature controls.\n")
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "render_e3_cifar_assets", "runs": len(rows), "estimator_control_rows": len(estimator_rows)}) + "\n")
    print(json.dumps({"runs": len(rows), "successful": len(successful), "estimator_control_rows": len(estimator_rows), "numerical_rows": len(numerical_rows), "output": str(output)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
