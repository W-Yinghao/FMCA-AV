#!/usr/bin/env python3
"""Aggregate completed dependence/localization outputs into E9 paper assets."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import random
import re
import statistics


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8"); temporary.replace(path)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values); position = probability * (len(ordered) - 1)
    lower = int(math.floor(position)); upper = int(math.ceil(position)); fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def hierarchical_interval(samples_by_run: list[list[float]], seed: int, replicates: int = 2000) -> tuple[object, object]:
    usable = [values for values in samples_by_run if values]
    if not usable:
        return "", ""
    generator = random.Random(seed); estimates = []
    for _ in range(replicates):
        selected_runs = [usable[generator.randrange(len(usable))] for _ in usable]
        selected_values = []
        for values in selected_runs:
            selected_values.extend(values[generator.randrange(len(values))] for _ in values)
        estimates.append(statistics.fmean(selected_values))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def main() -> int:
    output = Path("results/e9"); rows = []; per_run_samples: dict[str, dict[str, list[float]]] = {}
    localization_paths = list(Path("runs").glob("*/artifacts/localization.json"))
    localization_paths += list(Path("runs").glob("*/artifacts/supervised_localization.json"))
    for result_path in sorted(localization_paths):
        run_id = result_path.parents[1].name
        status_path = result_path.parents[1] / "status.json"
        if not status_path.is_file() or json.loads(status_path.read_text(encoding="utf-8")).get("state") != "SUCCEEDED": continue
        payload = json.loads(result_path.read_text(encoding="utf-8")); summary = payload.get("summary", {})
        preferred = "spectral" if "spectral" in summary else ("projector_energy" if "projector_energy" in summary else ("eigen_cam" if "eigen_cam" in summary else ""))
        metrics = summary.get(preferred, {}) if preferred else {}
        sample_metrics: dict[str, list[float]] = {}
        for record in payload.get("records", []):
            values = dict(dict(record).get("maps", {})).get(preferred, {}) if preferred else {}
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    sample_metrics.setdefault(str(key), []).append(float(value))
        per_run_samples[run_id] = sample_metrics
        request = json.loads((result_path.parents[1] / "request.json").read_text(encoding="utf-8"))
        name = str(request.get("name", "")); architecture = next((value for value in ("convnext_tiny", "vit_s_16", "vgg16_bn", "resnet50") if value.replace("_", "-") in name.replace("_", "-")), "")
        seed_match = re.search(r"(?:^|[-_])seed[-_]?([0-9]+)(?:[-_]|$)", name.lower())
        if not architecture and ("imagenet1k" in name or name.startswith("formal-e9-")): architecture = "resnet50"
        rows.append({
            "run_id": run_id, "name": name, "dataset": payload.get("dataset", ""), "method": payload.get("method", "fmca_av"),
            "seed": seed_match.group(1) if seed_match else "",
            "architecture": architecture, "map": preferred, "randomized": payload.get("randomize_backbone", False),
            "randomize_from_stage": payload.get("randomize_from_stage", ""),
            "samples": payload.get("samples", 0), "runtime_seconds": payload.get("runtime_seconds", ""),
            "peak_memory_mb": payload.get("peak_memory_mb", ""), "box_iou_top20": metrics.get("box_iou_top20", ""),
            "max_box_iou": metrics.get("max_box_iou", ""), "max_box_iou_quantile": metrics.get("max_box_iou_quantile", ""),
            "max_box_acc_iou50": metrics.get("max_box_acc_iou50", ""), "max_box_acc_quantile": metrics.get("max_box_acc_quantile", ""),
            "pointing_game": metrics.get("pointing_game", ""), "foreground_energy_ratio": metrics.get("foreground_energy_ratio", ""),
            "top20_mask_iou": metrics.get("top20_mask_iou", ""), "pixel_auprc": metrics.get("pixel_auprc", ""),
            "top_representation_cosine_drop": metrics.get("top_representation_cosine_drop", ""),
            "random_representation_cosine_drop": metrics.get("random_representation_cosine_drop", ""),
            "top_deletion_cosine_auc": metrics.get("top_deletion_cosine_auc", ""),
            "top_insertion_cosine_auc": metrics.get("top_insertion_cosine_auc", ""),
            "top_faithfulness_auc_gap": metrics.get("top_faithfulness_auc_gap", ""),
            "random_deletion_cosine_auc": metrics.get("random_deletion_cosine_auc", ""),
            "random_insertion_cosine_auc": metrics.get("random_insertion_cosine_auc", ""),
            "random_faithfulness_auc_gap": metrics.get("random_faithfulness_auc_gap", ""),
        })
    fields = ["run_id", "name", "dataset", "method", "seed", "architecture", "map", "randomized", "randomize_from_stage", "samples", "runtime_seconds", "peak_memory_mb", "box_iou_top20", "max_box_iou", "max_box_iou_quantile", "max_box_acc_iou50", "max_box_acc_quantile",
              "pointing_game", "foreground_energy_ratio", "top20_mask_iou", "pixel_auprc", "top_representation_cosine_drop", "random_representation_cosine_drop",
              "top_deletion_cosine_auc", "top_insertion_cosine_auc", "top_faithfulness_auc_gap",
              "random_deletion_cosine_auc", "random_insertion_cosine_auc", "random_faithfulness_auc_gap"]
    output.mkdir(parents=True, exist_ok=True); temporary = output / "localization_table.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(output / "localization_table.csv")
    metric_names = ("top20_mask_iou", "pixel_auprc", "pointing_game", "foreground_energy_ratio", "top_faithfulness_auc_gap")
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row["randomized"] or row["randomize_from_stage"] not in {"", None}:
            continue
        grouped.setdefault((str(row["dataset"]), str(row["method"]), str(row["architecture"]), str(row["map"])), []).append(row)
    summary_rows = []
    for key, values in sorted(grouped.items()):
        summary: dict[str, object] = {"dataset": key[0], "method": key[1], "architecture": key[2], "map": key[3], "runs": len(values)}
        for metric in metric_names:
            samples = [float(value[metric]) for value in values if value[metric] != ""]
            mean = statistics.fmean(samples) if samples else ""
            std = statistics.stdev(samples) if len(samples) > 1 else 0.0 if samples else ""
            summary[metric + "_mean"] = mean
            summary[metric + "_std"] = std
            summary[metric + "_ci95_half_width"] = 1.96 * float(std) / math.sqrt(len(samples)) if samples else ""
            bootstrap_low, bootstrap_high = hierarchical_interval(
                [per_run_samples.get(str(value["run_id"]), {}).get(metric, []) for value in values],
                seed=sum(ord(character) for character in "|".join(key) + metric),
            )
            summary[metric + "_hierarchical_bootstrap_ci95_low"] = bootstrap_low
            summary[metric + "_hierarchical_bootstrap_ci95_high"] = bootstrap_high
        summary_rows.append(summary)
    summary_fields = ["dataset", "method", "architecture", "map", "runs"] + [
        metric + suffix for metric in metric_names for suffix in (
            "_mean", "_std", "_ci95_half_width", "_hierarchical_bootstrap_ci95_low", "_hierarchical_bootstrap_ci95_high"
        )
    ]
    temporary = output / "localization_summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields); writer.writeheader(); writer.writerows(summary_rows)
    temporary.replace(output / "localization_summary.csv")
    plot_rows = [row for row in summary_rows if row["top20_mask_iou_mean"] != ""]
    width = 1200; height = max(360, 80 + 28 * len(plot_rows)); left = 390; chart = 740
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<style>.label{font:12px sans-serif}.title{font:600 16px sans-serif}.grid{stroke:#ddd}.bar{fill:#2855a6}</style>',
           '<text x="20" y="28" class="title">E9 dependence-map top-20% foreground IoU</text>']
    for tick in range(6):
        value = tick / 5; x = left + chart * value; svg.append(f'<line x1="{x}" y1="45" x2="{x}" y2="{height-20}" class="grid"/><text x="{x-8}" y="43" class="label">{value:.1f}</text>')
    for index, row in enumerate(plot_rows):
        y = 60 + 28 * index; value = float(row["top20_mask_iou_mean"]); ci = float(row["top20_mask_iou_ci95_half_width"])
        label = f'{row["dataset"]} | {row["method"]} | {row["architecture"] or "default"} | n={row["runs"]}'
        center = left + chart * value
        svg.append(f'<text x="20" y="{y+13}" class="label">{label}</text><rect x="{left}" y="{y}" width="{chart*value:.2f}" height="18" class="bar"/><line x1="{max(left, center-chart*ci):.2f}" y1="{y+9}" x2="{center+chart*ci:.2f}" y2="{y+9}" stroke="#111"/><text x="{center+5:.2f}" y="{y+13}" class="label">{value:.3f}</text>')
    svg.append("</svg>"); atomic_text(output / "localization_iou.svg", "".join(svg))
    composition_rows = []
    for result_path in sorted(Path("runs").glob("*/artifacts/cnn_composition_maps.json")):
        run_dir = result_path.parents[1]; status_path = run_dir / "status.json"
        if not status_path.is_file() or json.loads(status_path.read_text(encoding="utf-8")).get("state") != "SUCCEEDED": continue
        payload = json.loads(result_path.read_text(encoding="utf-8")); summary = dict(payload.get("summary", {}))
        request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
        composition_rows.append({
            "run_id": run_dir.name, "name": request.get("name", ""), "model_type": payload.get("model_type", ""),
            "backbone": payload.get("backbone", ""), "modes": payload.get("modes", ""),
            "calibration_samples": payload.get("calibration_samples", ""), "evaluation_samples": payload.get("evaluation_samples", ""),
            "runtime_seconds": payload.get("runtime_seconds", ""), "peak_memory_mb": payload.get("peak_memory_mb", ""),
            "rank_correlation": summary.get("rank_correlation", ""), "normalized_l2": summary.get("normalized_l2", ""),
            "top20_iou": summary.get("top20_iou", ""), "composition_assumption": payload.get("composition_assumption", ""),
        })
    composition_fields = ["run_id", "name", "model_type", "backbone", "modes", "calibration_samples", "evaluation_samples", "runtime_seconds", "peak_memory_mb",
                          "rank_correlation", "normalized_l2", "top20_iou", "composition_assumption"]
    temporary = output / "cnn_composition_table.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=composition_fields); writer.writeheader(); writer.writerows(composition_rows)
    temporary.replace(output / "cnn_composition_table.csv")
    caption = ("Quantitative E9 localization and faithfulness results from every successful harness run. "
               "The raw table retains negative and randomized-backbone controls and reports top/random deletion-insertion cosine AUC; "
               "the summary reports seed dispersion and a deterministic hierarchical bootstrap that resamples seeds and images. cnn_composition_table.csv reports direct-vs-recursive "
               "stage-operator rank correlation, normalized L2, and top-region IoU under the stated residual-stage assumption.\n")
    atomic_text(output / "localization_caption.txt", caption)
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "render_e9_assets", "runs": len(rows), "composition_runs": len(composition_rows)}) + "\n")
    print(json.dumps({"runs": len(rows), "composition_runs": len(composition_rows), "output": str(output)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
