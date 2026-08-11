#!/usr/bin/env python3
"""Aggregate completed CIFAR-10 FastSSL/FroSSL runs without mixing live results."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "postfix" / SCIENTIFIC_CORRECTNESS_VERSION / "external_multiview_baselines"


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def artifact(run_id: str, *names: str) -> dict[str, object]:
    directory = ROOT / "runs" / run_id / "artifacts"
    for name in names:
        path = directory / name
        if path.is_file():
            return read(path)
    raise FileNotFoundError(f"missing {names} in {run_id}")


def elapsed(left: object, right: object) -> float:
    return (datetime.fromisoformat(str(right)) - datetime.fromisoformat(str(left))).total_seconds()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--state-file", required=True)
    parser.add_argument("--output-subdir", default=""); args = parser.parse_args()
    output = OUTPUT / args.output_subdir if args.output_subdir else OUTPUT
    state = read(Path(args.state_file))
    if state.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError("refusing to aggregate another correctness version")
    records = dict(state["records"])
    required = [record for record in records.values() if record["kind"] not in {"smoke", "aggregate"}]
    if any(record.get("state") != "SUCCEEDED" for record in required):
        raise RuntimeError("all formal actions must succeed before aggregation")
    flops = {}
    for record in required:
        if record["kind"] == "flops":
            flops[(record["method"], int(record["views"]))] = artifact(str(record["run_id"]), "flops.json")
    rows = []
    for record in required:
        if record["kind"] != "train":
            continue
        method = str(record["method"]); views = int(record["views"]); seed_index = int(record["seed_index"])
        suffix = f":{method}:v{views}:s{seed_index}"
        related = {str(item["kind"]): item for item in required if str(item["key"]).endswith(suffix)}
        train = artifact(str(record["run_id"]), "train_result.json")
        train_status = read(ROOT / "runs" / str(record["run_id"]) / "status.json")
        probe = artifact(str(related["probe"]["run_id"]), "probe_result.json")
        knn = artifact(str(related["knn"]["run_id"]), "knn_result.json", "knn.json")
        diagnostics = artifact(str(related["diagnostics"]["run_id"]), "diagnostics.json")
        profile = flops[(method, views)]
        parents = float(train["encoded_views"]) / views
        rows.append({
            "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
            "method": method, "views": views, "seed_index": seed_index, "seed": record["seed"],
            "train_run": record["run_id"], "probe_run": related["probe"]["run_id"],
            "slurm_job_id": train_status["slurm_job_id"],
            "knn_run": related["knn"]["run_id"], "diagnostics_run": related["diagnostics"]["run_id"],
            "linear_probe_accuracy": probe["test_accuracy"], "knn_accuracy": knn["knn_accuracy"],
            "best_validation_ssl_score": train["best_validation_score"],
            "completed_optimizer_steps": train["completed_optimizer_steps"],
            "training_duration_seconds": train["training_duration_seconds"], "gpu_hours": train["gpu_hours"],
            "harness_wall_clock_seconds": elapsed(train_status["start_time"], train_status["end_time"]),
            "queue_wait_seconds": elapsed(train_status["created_at"], train_status["start_time"]),
            "encoded_views": train["encoded_views"], "encoded_views_per_second": train["encoded_views_per_second"],
            "trainable_parameters": train["trainable_parameters"], "peak_memory_mb_per_rank": train["peak_memory_mb_per_rank"],
            "gpu_name": train["gpu_name"], "supported_flops_per_parent": profile["flops_per_parent"],
            "estimated_total_supported_operator_flops": float(profile["flops_per_parent"]) * parents,
            "backbone_effective_rank": diagnostics["backbone"]["effective_rank"],
            "backbone_normalized_effective_rank": diagnostics["backbone"]["normalized_effective_rank"],
            "backbone_covariance_trace": diagnostics["backbone"]["covariance_trace"],
            "backbone_mean_abs_offdiag_correlation": diagnostics["backbone"]["mean_absolute_off_diagonal_correlation"],
            "backbone_collapsed_dimension_fraction": diagnostics["backbone"]["collapsed_dimension_fraction_std_lt_1e-2"],
            "projector_effective_rank": diagnostics["projector"]["effective_rank"],
            "projector_normalized_effective_rank": diagnostics["projector"]["normalized_effective_rank"],
            "projector_covariance_trace": diagnostics["projector"]["covariance_trace"],
            "projector_mean_abs_offdiag_correlation": diagnostics["projector"]["mean_absolute_off_diagonal_correlation"],
            "projector_collapsed_dimension_fraction": diagnostics["projector"]["collapsed_dimension_fraction_std_lt_1e-2"],
        })
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    csv_path = output / "run_matrix.csv"; temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    temporary.replace(csv_path)
    summaries = []
    official = {
        ("fastssl_barlow_twins", 2): 86.43,
        ("fastssl_barlow_twins", 8): 92.71,
        ("frossl", 2): 92.8,
    }
    for method in sorted({str(row["method"]) for row in rows}):
        for views in sorted({int(row["views"]) for row in rows if row["method"] == method}):
            group = [row for row in rows if row["method"] == method and row["views"] == views]
            values = [100 * float(row["linear_probe_accuracy"]) for row in group]
            target = official.get((method, views))
            summaries.append({
                "method": method, "views": views, "seeds": len(group),
                "linear_probe_mean_percent": statistics.fmean(values),
                "linear_probe_sample_std_percent": statistics.stdev(values),
                "knn_mean_percent": 100 * statistics.fmean(float(row["knn_accuracy"]) for row in group),
                "gpu_hours_total": sum(float(row["gpu_hours"]) for row in group),
                "official_paper_cifar10_percent": target,
                "difference_from_official_percentage_points": statistics.fmean(values) - target if target is not None else None,
            })
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "scope": "CIFAR-10 external multi-view/multi-layer SSL baselines; state-file scoped",
        "source_lock": read(ROOT / "configs" / "external_baseline_sources.json"),
        "rows": rows, "summaries": summaries,
        "comparison_notes": {
            "fastssl_barlow_twins": "paper Table 7, projector 256, 100 pretraining epochs",
            "fastssl_vicreg": "paper plots but does not tabulate an exact CIFAR-10 2/8-view value",
            "frossl": "paper Table 5 reports CIFAR-10 only for 2 views and states 4/8-view gains were negligible",
            "hai_simsiam": "faithful reimplementation; the HAI paper reports ImageNet/ImageNet-100, not CIFAR-10",
        },
    }
    atomic_json(output / "results.json", payload)
    lines = [
        "# CIFAR-10 external multi-view SSL baselines", "",
        f"Scientific correctness version: `{SCIENTIFIC_CORRECTNESS_VERSION}`.", "",
        "| Method | Views | Seeds | Linear probe (%) | kNN (%) | GPU-hours | Paper (%) | Delta (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        target = "" if item["official_paper_cifar10_percent"] is None else f"{item['official_paper_cifar10_percent']:.2f}"
        delta = "" if item["difference_from_official_percentage_points"] is None else f"{item['difference_from_official_percentage_points']:.2f}"
        lines.append(
            f"| {item['method']} | {item['views']} | {item['seeds']} | "
            f"{item['linear_probe_mean_percent']:.2f} | {item['knn_mean_percent']:.2f} | "
            f"{item['gpu_hours_total']:.2f} | {target} | {delta} |"
        )
    lines.extend(["", "Full run-level cost and collapse/covariance diagnostics are in `run_matrix.csv`.", ""])
    report = output / "README.md"; temporary_report = report.with_suffix(".md.tmp")
    temporary_report.write_text("\n".join(lines), encoding="utf-8"); temporary_report.replace(report)
    print(json.dumps({"rows": len(rows), "summaries": len(summaries), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
