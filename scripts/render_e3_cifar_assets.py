#!/usr/bin/env python3
"""Render CIFAR numerical/objective ablation assets from completed E3 runs."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics


SOURCES = (
    Path("runs/20260807-062417_launch-e3-cifar-numerics-wave/artifacts/submitted.json"),
    Path("runs/20260807-073411_recover-interrupted-e3-tsd-coco-v2/artifacts/e3_submitted.json"),
)


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
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        row: dict[str, object] = {"dataset": record.get("dataset", "cifar10"), "tag": tag, "seed": record["seed"], "run_id": run_id, "state": status["state"],
                                  "override_json": json.dumps(record["override"], sort_keys=True, separators=(",", ":")),
                                  "duration_seconds": "", "best_validation_score": "", "trace": "", "numerical_rank": "",
                                  "effective_rank": "", "condition_number": "", "minimum_eigenvalue": "", "maximum_eigenvalue": ""}
        if status.get("start_time") and status.get("end_time"):
            from datetime import datetime
            row["duration_seconds"] = (datetime.fromisoformat(status["end_time"]) - datetime.fromisoformat(status["start_time"])).total_seconds()
        train_path = run_dir / "artifacts" / "train_result.json"; evaluation_path = run_dir / "artifacts" / "evaluation.json"
        if status["state"] == "SUCCEEDED" and train_path.is_file():
            row["best_validation_score"] = json.loads(train_path.read_text(encoding="utf-8")).get("best_validation_score", "")
        if status["state"] == "SUCCEEDED" and evaluation_path.is_file():
            values = json.loads(evaluation_path.read_text(encoding="utf-8")).get("test_empirical_eigenvalues", [])
            row.update(spectral_metrics(values))
        rows.append(row)
    output = Path("results/e3"); output.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "tag", "seed", "run_id", "state", "override_json", "duration_seconds", "best_validation_score", "trace",
              "numerical_rank", "effective_rank", "condition_number", "minimum_eigenvalue", "maximum_eigenvalue"]
    temporary = output / "cifar_numerics_table.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(output / "cifar_numerics_table.csv")
    estimator_rows = []
    for control_path in sorted(Path("runs").glob("*/artifacts/e3_estimator_controls.json")):
        status_path = control_path.parents[1] / "status.json"
        if not status_path.is_file() or json.loads(status_path.read_text(encoding="utf-8")).get("state") != "SUCCEEDED":
            continue
        run_id = control_path.parents[1].name
        for record in json.loads(control_path.read_text(encoding="utf-8")).get("records", []):
            estimator_rows.append({"run_id": run_id, **dict(record)})
    estimator_fields = sorted({key for row in estimator_rows for key in row}) if estimator_rows else ["run_id"]
    temporary = output / "estimator_controls.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=estimator_fields); writer.writeheader(); writer.writerows(estimator_rows)
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
        writer = csv.DictWriter(handle, fieldnames=control_summary_fields); writer.writeheader(); writer.writerows(control_summaries)
    temporary.replace(output / "estimator_controls_summary.csv")
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
                "E3 CIFAR-10 one-factor/fractional ablations and three-seed ImageNet-100 reference/log-det/stable/stress rechecks. Failed settings remain in the CSV; bars show the frozen best validation dependence score, while available held-out evaluations provide rank and conditioning diagnostics. The estimator-control tables add 20-seed Gaussian/finite-channel centered, whitening, adaptive-ridge, objective, precision, batch/sample and post-hoc spectral controls with failure rates and 95% confidence intervals.\n")
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "render_e3_cifar_assets", "runs": len(rows), "estimator_control_rows": len(estimator_rows)}) + "\n")
    print(json.dumps({"runs": len(rows), "successful": len(successful), "estimator_control_rows": len(estimator_rows), "output": str(output)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
