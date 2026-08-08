#!/usr/bin/env python3
"""Aggregate E6 low-label, transfer, detection, and segmentation experiments."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re


METHODS = (
    "fmca_av_matched_head", "hfmca_style", "regular_fmca", "spectral_contrastive",
    "barlow_twins", "moco_v2", "fastsiam", "simclr", "vicreg", "byol", "dino",
    "dcca", "vamp2", "fmca_av",
)
DATASETS = ("imagenet100", "imagenet1k", "cifar100", "cifar10", "stl10", "tinyimagenet200", "voc2007", "coco2017")


def read(path: Path) -> dict[str, object]: return json.loads(path.read_text(encoding="utf-8"))


def labels(name: str) -> tuple[str, str, str]:
    normalized = name.lower().replace("-", "_")
    dataset = next((value for value in DATASETS if value in normalized), "")
    method = next((value for value in METHODS if value in normalized), "fmca_av" if "fmca" in normalized or "gmean" in normalized else ("supervised" if "supervised" in normalized else ""))
    seed_match = re.search(r"(?:^|[-_])seed[-_]?([0-9]+)(?:[-_]|$)", name.lower())
    return dataset, method, seed_match.group(1) if seed_match else ""


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8"); temporary.replace(path)


def main() -> int:
    jobs = read(Path("harness/state/jobs.json"))["jobs"]; rows: list[dict[str, object]] = []
    for run_id, job_value in jobs.items():
        job = dict(job_value); name = str(job.get("name", "")); run_dir = Path("runs") / run_id
        if job.get("state") != "SUCCEEDED": continue
        dataset, method, seed = labels(name)
        for filename, protocol in (("probe_result.json", "linear_probe"), ("finetune_result.json", "fine_tune")):
            path = run_dir / "artifacts" / filename
            if not path.is_file(): continue
            value = read(path)
            rows.append({"run_id": run_id, "name": name, "dataset": dataset, "method": method, "seed": seed, "protocol": protocol,
                         "label_fraction": value.get("label_fraction", ""), "primary_metric": "accuracy",
                         "primary_value": value.get("test_accuracy", ""), "secondary_metric": "top5_accuracy",
                         "secondary_value": value.get("test_top5_accuracy", ""), "samples": "", "task": "", "evaluation_protocol": ""})
        path = run_dir / "artifacts" / "voc2007_multilabel.json"
        if path.is_file():
            value = read(path); rows.append({"run_id": run_id, "name": name, "dataset": "voc2007", "method": method, "seed": seed,
                "protocol": "frozen_multilabel_probe", "label_fraction": 1.0, "primary_metric": "mAP",
                "primary_value": value.get("test_map", ""), "secondary_metric": "validation_mAP",
                "secondary_value": value.get("best_validation_map", ""), "samples": "", "task": "classification", "evaluation_protocol": "multilabel_average_precision"})
        path = run_dir / "artifacts" / "coco_transfer.json"
        if path.is_file():
            value = read(path); task = str(value.get("task", "")); key = "segm_AP" if task == "instance_segmentation" else "bbox_AP"
            secondary = "segm_AP50" if task == "instance_segmentation" else "bbox_AP50"
            rows.append({"run_id": run_id, "name": name, "dataset": "coco2017", "method": method, "seed": seed, "protocol": "transfer",
                "label_fraction": "", "primary_metric": key, "primary_value": value.get(key, ""),
                "secondary_metric": secondary, "secondary_value": value.get(secondary, ""),
                "samples": value.get("evaluated_images", ""), "task": task,
                "evaluation_protocol": value.get("evaluation_protocol", "legacy_custom_coco_style_101_point")})
        path = run_dir / "artifacts" / "voc_detection.json"
        if path.is_file():
            value = read(path); rows.append({"run_id": run_id, "name": name, "dataset": "voc2007+2012", "method": method, "seed": seed,
                "protocol": "transfer", "label_fraction": "", "primary_metric": "bbox_AP", "primary_value": value.get("bbox_AP", ""),
                "secondary_metric": "bbox_AP50", "secondary_value": value.get("bbox_AP50", ""),
                "samples": value.get("evaluated_images", ""), "task": "detection",
                "evaluation_protocol": value.get("evaluation_protocol", "custom_coco_style_101_point_voc_macro")})
    fields = ["run_id", "name", "dataset", "method", "seed", "protocol", "label_fraction", "primary_metric", "primary_value", "secondary_metric", "secondary_value", "samples", "task", "evaluation_protocol"]
    output = Path("results/e6"); write_csv(output / "generalization_transfer_table.csv", rows, fields)
    plotted = [row for row in rows if row["primary_value"] != ""]
    width = 1250; height = max(340, 80 + 26 * len(plotted)); left = 480; chart = 690
    maximum = max((float(row["primary_value"]) for row in plotted), default=1.0)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<style>.label{font:11px sans-serif}.title{font:600 16px sans-serif}.bar{fill:#2855a6}</style>',
           '<text x="20" y="28" class="title">E6 low-label and transfer primary metrics</text>']
    for index, row in enumerate(plotted):
        y = 56 + 26 * index; value = float(row["primary_value"]); length = chart * value / max(maximum, 1e-12)
        label = f'{row["dataset"]} | {row["method"]} | {row["protocol"]} | labels={row["label_fraction"] or "all"} | {row["primary_metric"]}'
        svg.append(f'<text x="20" y="{y+12}" class="label">{label}</text><rect x="{left}" y="{y}" width="{length:.2f}" height="16" class="bar"/><text x="{left+length+5:.2f}" y="{y+12}" class="label">{value:.4f}</text>')
    svg.append("</svg>"); atomic_text(output / "generalization_transfer.svg", "".join(svg))
    atomic_text(output / "generalization_transfer_caption.txt", "E6 low-label frozen-probe/fine-tuning and VOC/COCO transfer results. Each row names its evaluation protocol: future formal COCO bbox/segmentation AP uses official pycocotools COCOeval, while retained legacy smoke rows and VOC detection are explicitly labeled as 101-point custom protocols. Short-smoke zero or negative results are retained rather than promoted as successful transfer. Claim IDs: E6/C3.\n")
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "render_e6_generalization_assets", "rows": len(rows)}) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
