#!/usr/bin/env python3
"""Aggregate CIFAR-C and ImageNet-C/R/A evaluations into E6 assets."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import re
import statistics


METHODS = (
    "fmca_av_matched_head", "hfmca_style", "regular_fmca", "spectral_contrastive",
    "barlow_twins", "moco_v2", "fastsiam", "simclr", "vicreg", "byol", "dino",
    "dcca", "vamp2", "fmca_av",
)


def labels(name: str) -> tuple[str, str]:
    normalized = name.lower().replace("-", "_")
    method = next((value for value in METHODS if value in normalized), "")
    match = re.search(r"(?:^|[-_])seed[-_]?([0-9]+)(?:[-_]|$)", name.lower())
    return method, match.group(1) if match else ""


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8"); temporary.replace(path)


def main() -> int:
    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for path in sorted(Path("runs").glob("*/artifacts/*.json")):
        try: payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError): continue
        run_dir = path.parents[1]; status_path = run_dir / "status.json"
        if not status_path.is_file() or json.loads(status_path.read_text(encoding="utf-8")).get("state") != "SUCCEEDED": continue
        request = json.loads((run_dir / "request.json").read_text(encoding="utf-8")); name = str(request.get("name", ""))
        method, seed = labels(name)
        common = {"run_id": run_dir.name, "name": name, "file": path.name, "mean_accuracy": "", "mce": "",
                  "relative_mce": "", "clean_accuracy": "", "ece": "", "samples": "",
                  "method": method, "seed": seed}
        if "corruptions" in payload and "mean_corruption_accuracy" in payload:
            dataset = "cifar100" if "cifar100" in name else "cifar10"
            mean = float(payload["mean_corruption_accuracy"])
            rows.append({**common, "dataset": dataset, "suite": f"{dataset}-c", "mean_accuracy": mean,
                         "mce": payload.get("mean_corruption_error_percent", 100.0 * (1.0 - mean)),
                         "relative_mce": payload.get("relative_mean_corruption_error_percent", ""),
                         "clean_accuracy": payload.get("clean_accuracy", "")})
            corruptions = payload.get("corruptions", {})
            if isinstance(corruptions, dict):
                for corruption, severities in sorted(corruptions.items()):
                    if isinstance(severities, dict):
                        for severity, accuracy in sorted(severities.items(), key=lambda item: int(item[0])):
                            detail_rows.append({"run_id": run_dir.name, "name": name, "dataset": dataset,
                                                "suite": f"{dataset}-c", "corruption": corruption,
                                                "severity": severity, "accuracy": accuracy, "family": ""})
        if "imagenet_c" in payload:
            value = payload["imagenet_c"]
            rows.append({**common, "dataset": payload.get("dataset", "imagenet1k"), "suite": "imagenet-c",
                         "mean_accuracy": value.get("mean_corruption_accuracy", ""), "mce": value.get("mce", ""),
                         "relative_mce": value.get("relative_mce", ""),
                         "clean_accuracy": value.get("clean", {}).get("top1_accuracy", "")})
            corruptions = value.get("corruptions", {})
            if isinstance(corruptions, dict):
                for corruption, corruption_value in sorted(corruptions.items()):
                    family = corruption_value.get("family", "") if isinstance(corruption_value, dict) else ""
                    severities = corruption_value.get("severities", corruption_value) if isinstance(corruption_value, dict) else {}
                    if isinstance(severities, dict):
                        for severity, accuracy in sorted(severities.items(), key=lambda item: str(item[0])):
                            if isinstance(accuracy, dict): accuracy = accuracy.get("top1_accuracy", accuracy.get("accuracy", ""))
                            detail_rows.append({"run_id": run_dir.name, "name": name, "dataset": payload.get("dataset", "imagenet1k"),
                                                "suite": "imagenet-c", "corruption": corruption,
                                                "severity": severity, "accuracy": accuracy, "family": family})
        for key, suite in (("imagenet_r", "imagenet-r"), ("imagenet_a", "imagenet-a")):
            if key in payload:
                value = payload[key]
                rows.append({**common, "dataset": payload.get("dataset", "imagenet1k"), "suite": suite,
                             "mean_accuracy": value.get("top1_accuracy", ""), "ece": value.get("ece_15_bin", ""),
                             "samples": value.get("samples", "")})
    output = Path("results/e6"); output.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "name", "file", "dataset", "method", "seed", "suite", "mean_accuracy", "mce", "relative_mce", "clean_accuracy", "ece", "samples"]
    temporary = output / "robustness_table.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(output / "robustness_table.csv")
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row["mean_accuracy"] == "":
            continue
        grouped.setdefault((str(row["dataset"]), str(row["method"]), str(row["suite"])), []).append(row)
    summary_rows = []
    for (dataset, method, suite), values in sorted(grouped.items()):
        accuracies = [float(value["mean_accuracy"]) for value in values]
        mean = statistics.fmean(accuracies)
        std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        summary_rows.append({
            "dataset": dataset, "method": method, "suite": suite, "runs": len(accuracies),
            "mean_accuracy": mean, "std_accuracy": std,
            "ci95_half_width": 1.96 * std / math.sqrt(len(accuracies)),
        })
    summary_fields = ["dataset", "method", "suite", "runs", "mean_accuracy", "std_accuracy", "ci95_half_width"]
    temporary = output / "robustness_summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields); writer.writeheader(); writer.writerows(summary_rows)
    temporary.replace(output / "robustness_summary.csv")
    detail_fields = ["run_id", "name", "dataset", "suite", "corruption", "family", "severity", "accuracy"]
    temporary = output / "robustness_per_corruption.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields); writer.writeheader(); writer.writerows(detail_rows)
    temporary.replace(output / "robustness_per_corruption.csv")
    plotted = summary_rows
    width = 1200; height = max(320, 75 + 28 * len(plotted)); left = 440; chart = 680
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<style>.label{font:11px sans-serif}.title{font:600 16px sans-serif}.grid{stroke:#ddd}.bar{fill:#2855a6}</style>',
           '<text x="20" y="28" class="title">E6 corruption and OOD accuracy</text>']
    for tick in range(6):
        x = left + chart * tick / 5; svg.append(f'<line x1="{x}" y1="45" x2="{x}" y2="{height-15}" class="grid"/><text x="{x-7}" y="43" class="label">{tick/5:.1f}</text>')
    for index, row in enumerate(plotted):
        y = 58 + 28 * index; value = float(row["mean_accuracy"]); label = f'{row["suite"]} | {row["method"] or "unclassified"} | n={row["runs"]}'
        center = left + chart * value; ci = chart * float(row["ci95_half_width"])
        svg.append(f'<text x="20" y="{y+13}" class="label">{label[:65]}</text><rect x="{left}" y="{y}" width="{chart*value:.2f}" height="18" class="bar"/><line x1="{max(left, center-ci):.2f}" y1="{y+9}" x2="{center+ci:.2f}" y2="{y+9}" stroke="#111"/><text x="{center+5:.2f}" y="{y+13}" class="label">{value:.3f}</text>')
    svg.append("</svg>"); atomic_text(output / "robustness_accuracy.svg", "".join(svg))
    atomic_text(output / "robustness_caption.txt",
                "E6 clean/corruption/OOD results. The summary and figure report mean and normal-approximation 95% CI over independent seeds. ImageNet-C mCE uses the canonical 15-corruption AlexNet normalization; four extra corruptions are retained separately. Legacy CIFAR-C runs without a stored clean score report unnormalized mean corruption error only.\n")
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "render_e6_robustness_assets", "rows": len(rows), "detail_rows": len(detail_rows)}) + "\n")
    print(json.dumps({"rows": len(rows), "detail_rows": len(detail_rows), "output": str(output)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
