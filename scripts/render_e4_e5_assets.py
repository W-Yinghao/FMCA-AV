#!/usr/bin/env python3
"""Build E4 architecture and E5 matched-SSL tables/figures from harness artifacts."""

from __future__ import annotations

import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import statistics


METHODS = (
    "spectral_contrastive", "barlow_twins", "regular_fmca", "hfmca_style",
    "moco_v2", "fastsiam", "simclr", "vicreg", "byol", "dino", "dcca",
    "vamp2", "fmca_av_matched_head", "fmca_av_deepsets", "fmca_av",
)
DATASETS = ("tinyimagenet200", "imagenet100", "imagenet1k", "cifar100", "cifar10", "stl10")
SOURCE_PATTERN = re.compile(r"/runs/([^/]+)/artifacts/")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def duration_seconds(status: dict[str, object]) -> float | str:
    if not status.get("start_time") or not status.get("end_time"): return ""
    return (datetime.fromisoformat(str(status["end_time"])) - datetime.fromisoformat(str(status["start_time"]))).total_seconds()


def labels(name: str, request: dict[str, object]) -> tuple[str, str, int | str, str, str]:
    normalized = name.lower().replace("-", "_")
    dataset = next((value for value in DATASETS if value in normalized), "")
    method = next((value for value in METHODS if value in normalized), "fmca_av" if "fmca" in normalized else "")
    view_match = re.search(r"(?:^|[-_])(?:v|m)([0-9]+)(?:[-_]|$)", name.lower())
    views: int | str = int(view_match.group(1)) if view_match else ""
    architecture = next((value for value in ("convnext_tiny", "vit_s_16", "vgg16_bn", "resnet50", "resnet18") if value in normalized), "")
    aggregation = next((value for value in ("deepsets", "concat", "mean", "first", "raw") if value in normalized), "")
    for item in request.get("final_command", []):
        text = str(item)
        if not text.startswith("FMCA_CONFIG_OVERRIDES="): continue
        try: override = json.loads(text.split("=", 1)[1])
        except json.JSONDecodeError: continue
        experiment = override.get("experiment", {})
        model = override.get("model", {}); data = override.get("data", {})
        method = str(experiment.get("method", method)); views = data.get("num_views", views)
        architecture = str(model.get("backbone", architecture)); aggregation = str(model.get("parent_aggregation", aggregation))
    return dataset, method, views, architecture, aggregation


def source_run(payload: dict[str, object]) -> str:
    value = str(payload.get("source_checkpoint") or payload.get("checkpoint") or "")
    match = SOURCE_PATTERN.search(value)
    return match.group(1) if match else ""


def seed_label(name: str) -> str:
    match = re.search(r"(?:^|[-_])seed[-_]?([0-9]+)(?:[-_]|$)", name.lower())
    return match.group(1) if match else ""


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def group_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[field] for field in ("dataset", "method", "views", "architecture", "aggregation", "protocol"))
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        metrics = [float(value["accuracy"]) for value in values if value["accuracy"] != ""]
        mean = statistics.fmean(metrics) if metrics else ""; std = statistics.stdev(metrics) if len(metrics) > 1 else 0.0 if metrics else ""
        ci = 1.96 * float(std) / math.sqrt(len(metrics)) if metrics else ""
        gpu_hours = [float(value["source_gpu_hours"]) for value in values if value["source_gpu_hours"] != ""]
        output.append({
            "dataset": key[0], "method": key[1], "views": key[2], "architecture": key[3],
            "aggregation": key[4], "protocol": key[5], "runs": len(values), "successful_metrics": len(metrics),
            "accuracy_mean": mean, "accuracy_std": std, "accuracy_ci95_half_width": ci,
            "source_gpu_hours_mean": statistics.fmean(gpu_hours) if gpu_hours else "",
        })
    return output


def svg_bars(rows: list[dict[str, object]]) -> str:
    plotted = [row for row in rows if row["accuracy_mean"] != ""]
    width = 1400; height = max(360, 80 + len(plotted) * 24); left = 520; chart = 800
    values = [float(row["accuracy_mean"]) for row in plotted]; maximum = max(values, default=1.0)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>.label{font:11px sans-serif}.title{font:600 16px sans-serif}.bar{fill:#2855a6}.ci{stroke:#111;stroke-width:1.2}</style>',
             '<text x="20" y="28" class="title">E5 matched-view frozen-representation accuracy (mean and 95% CI)</text>']
    for index, row in enumerate(plotted):
        y = 55 + 24 * index; mean = float(row["accuracy_mean"]); ci = float(row["accuracy_ci95_half_width"])
        label = f'{row["dataset"]} | {row["method"]} | V={row["views"]} | {row["architecture"] or "default"} | {row["protocol"]}'
        length = chart * mean / max(maximum, 1e-12); low = left + chart * max(0.0, mean-ci) / max(maximum, 1e-12); high = left + chart * (mean+ci) / max(maximum, 1e-12)
        parts.append(f'<text x="20" y="{y+12}" class="label">{label}</text><rect x="{left}" y="{y}" width="{length:.2f}" height="15" class="bar"/><line x1="{low:.2f}" y1="{y+7.5}" x2="{high:.2f}" y2="{y+7.5}" class="ci"/><text x="{left+length+5:.2f}" y="{y+12}" class="label">{mean:.4f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def main() -> int:
    jobs = read_json(Path("harness/state/jobs.json"))["jobs"]
    rows: list[dict[str, object]] = []
    for run_id, job_value in jobs.items():
        job = dict(job_value); run_dir = Path("runs") / run_id
        if job.get("state") != "SUCCEEDED" or not (run_dir / "request.json").is_file(): continue
        request = read_json(run_dir / "request.json"); name = str(request.get("name", job.get("name", "")))
        dataset, method, views, architecture, aggregation = labels(name, request)
        for filename, protocol, metric in (("probe_result.json", "linear_probe", "test_accuracy"), ("knn.json", "knn", "knn_accuracy")):
            path = run_dir / "artifacts" / filename
            if not path.is_file(): continue
            payload = read_json(path); source = source_run(payload); source_job = dict(jobs.get(source, {}))
            source_duration = duration_seconds(source_job) if source_job else ""; source_gpus = int(source_job.get("requested_gpus", 0)) if source_job else 0
            rows.append({
                "run_id": run_id, "source_run": source, "name": name, "dataset": dataset, "method": method,
                "views": views, "architecture": architecture, "aggregation": aggregation, "protocol": protocol,
                "seed": seed_label(name),
                "accuracy": payload.get(metric, ""), "source_duration_seconds": source_duration,
                "source_gpu_hours": float(source_duration) * source_gpus / 3600.0 if source_duration != "" else "",
            })
    fields = ["run_id", "source_run", "name", "dataset", "method", "views", "architecture", "aggregation", "protocol", "seed", "accuracy", "source_duration_seconds", "source_gpu_hours"]
    grouped = group_rows(rows); grouped_fields = ["dataset", "method", "views", "architecture", "aggregation", "protocol", "runs", "successful_metrics", "accuracy_mean", "accuracy_std", "accuracy_ci95_half_width", "source_gpu_hours_mean"]
    output = Path("results/e5"); write_csv(output / "matched_ssl_runs.csv", rows, fields); write_csv(output / "matched_ssl_summary.csv", grouped, grouped_fields)
    atomic_text(output / "matched_ssl_accuracy.svg", svg_bars(grouped))
    atomic_text(output / "matched_ssl_caption.txt", "E5 matched-view frozen linear-probe and weighted-kNN results. Error bars are normal-approximation 95% confidence intervals over frozen confirmatory seeds; all successful screening and formal runs remain in the raw table. Claim IDs: E5/C3.\n")
    e4_rows = []
    for run_id, job_value in jobs.items():
        job = dict(job_value); name = str(job.get("name", ""))
        if job.get("state") != "SUCCEEDED" or not name.startswith("e4-cifar10-"): continue
        result_path = Path("runs") / run_id / "artifacts" / "train_result.json"
        if not result_path.is_file(): continue
        request = read_json(Path("runs") / run_id / "request.json"); dataset, method, views, architecture, aggregation = labels(name, request)
        result = read_json(result_path)
        e4_rows.append({"run_id": run_id, "name": name, "dataset": dataset, "views": views, "architecture": architecture,
                        "aggregation": aggregation, "best_validation_score": result.get("best_validation_score", ""),
                        "parameters": result.get("total_parameters", ""), "duration_seconds": duration_seconds(job)})
    e4_fields = ["run_id", "name", "dataset", "views", "architecture", "aggregation", "best_validation_score", "parameters", "duration_seconds"]
    write_csv(Path("results/e4/aggregation_ablation.csv"), e4_rows, e4_fields)
    atomic_text(Path("results/e4/aggregation_caption.txt"), "E4 parent aggregation, head-sharing, gradient-stop, and feature-source ablations. The CSV retains every successful run and its frozen validation dependence score, parameter count, and runtime. Claim IDs: E4/C2/C3.\n")
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "render_e4_e5_assets", "e4_runs": len(e4_rows), "e5_rows": len(rows)}) + "\n")
    print(json.dumps({"e4_runs": len(e4_rows), "e5_rows": len(rows), "e5_groups": len(grouped)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
