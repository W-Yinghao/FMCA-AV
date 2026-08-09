#!/usr/bin/env python3
"""Aggregate TSD severity sweeps and downstream utility probes."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import re
import statistics

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


PATTERN = re.compile(r"tsd-(cifar10|cifar100|imagenet100)-([a-z]+)-level([0-9]+)-seed([0-9]+)")
RESULTS_ROOT = Path(os.environ.get(
    "FMCA_RESULTS_ROOT", f"results/postfix/{SCIENTIFIC_CORRECTNESS_VERSION}",
))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def has_postfix_train_result(run_id: str) -> bool:
    path = Path("runs") / run_id / "artifacts" / "train_result.json"
    return path.is_file() and read_json(path).get("scientific_correctness_version") == SCIENTIFIC_CORRECTNESS_VERSION


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8"); temporary.replace(path)


def main() -> int:
    jobs = json.loads(Path("harness/state/jobs.json").read_text(encoding="utf-8"))["jobs"]
    source_rows: dict[str, dict[str, object]] = {}
    probe_by_source: dict[str, tuple[str, object]] = {}
    for run_id, job in jobs.items():
        name = str(job.get("name", "")); match = PATTERN.search(name)
        if match and "utility-linear-probe" not in name:
            dataset, channel, level, seed_index = match.groups(); run_dir = Path("runs") / run_id
            if not has_postfix_train_result(run_id): continue
            calibration_path = run_dir / "artifacts" / "calibration.json"
            evaluation_path = run_dir / "artifacts" / "evaluation.json"
            calibration_score: object = ""; heldout_tsd: object = ""; gap: object = ""; clipped_modes: object = ""
            if calibration_path.is_file():
                calibration_score = json.loads(calibration_path.read_text(encoding="utf-8")).get("logdet_score", "")
            if evaluation_path.is_file():
                evaluation = read_json(evaluation_path)
                eigenvalues = evaluation.get("test_empirical_eigenvalues", []) if evaluation.get("scientific_correctness_version") == SCIENTIFIC_CORRECTNESS_VERSION else []
                if eigenvalues:
                    clipped_modes = sum(float(value) >= 1.0 - 1e-7 for value in eigenvalues)
                    heldout_tsd = sum(-math.log1p(-min(max(float(value), 0.0), 1.0 - 1e-7)) for value in eigenvalues)
            if calibration_score != "" and heldout_tsd != "": gap = float(calibration_score) - float(heldout_tsd)
            source_rows[run_id] = {"dataset": dataset, "channel": channel, "level": int(level), "seed_index": int(seed_index),
                                   "source_run": run_id, "source_state": job.get("state", ""), "dependence_score": heldout_tsd,
                                   "calibration_tsd": calibration_score, "heldout_tsd": heldout_tsd,
                                   "calibration_test_gap": gap, "heldout_clipped_mode_count": clipped_modes,
                                   "probe_run": "", "probe_state": "", "probe_accuracy": ""}
        if name.endswith("-utility-linear-probe"):
            source_run = name[:-len("-utility-linear-probe")]; probe_path = Path("runs") / run_id / "artifacts" / "probe_result.json"
            accuracy: object = ""
            if probe_path.is_file(): accuracy = json.loads(probe_path.read_text(encoding="utf-8")).get("test_accuracy", "")
            probe_by_source[source_run] = (run_id, accuracy)
    for source_run, row in source_rows.items():
        if source_run in probe_by_source:
            probe_run, accuracy = probe_by_source[source_run]; row["probe_run"] = probe_run
            row["probe_state"] = jobs[probe_run].get("state", ""); row["probe_accuracy"] = accuracy
    rows = sorted(source_rows.values(), key=lambda row: (str(row["dataset"]), str(row["channel"]), int(row["level"]), int(row["seed_index"])))
    output = RESULTS_ROOT / "e7"; output.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "channel", "level", "seed_index", "source_run", "source_state",
              "dependence_score", "calibration_tsd", "heldout_tsd", "calibration_test_gap",
              "heldout_clipped_mode_count", "probe_run", "probe_state", "probe_accuracy"]
    temporary = output / "tsd_utility_table.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    temporary.replace(output / "tsd_utility_table.csv")
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows: groups.setdefault((str(row["dataset"]), str(row["channel"]), int(row["level"])), []).append(row)
    means = []
    for (dataset, channel, level), group in sorted(groups.items()):
        scores = [float(row["dependence_score"]) for row in group if row["dependence_score"] != ""]
        accuracies = [float(row["probe_accuracy"]) for row in group if row["probe_accuracy"] != ""]
        score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0 if scores else ""
        accuracy_std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0 if accuracies else ""
        gaps = [float(row["calibration_test_gap"]) for row in group if row["calibration_test_gap"] != ""]
        clipped = [float(row["heldout_clipped_mode_count"]) for row in group if row["heldout_clipped_mode_count"] != ""]
        means.append({"dataset": dataset, "channel": channel, "level": level,
                      "dependence_score_mean": statistics.fmean(scores) if scores else "",
                      "dependence_score_std": score_std,
                      "dependence_score_ci95_half_width": 1.96 * float(score_std) / math.sqrt(len(scores)) if scores else "",
                      "probe_accuracy_mean": statistics.fmean(accuracies) if accuracies else "",
                      "probe_accuracy_std": accuracy_std,
                      "probe_accuracy_ci95_half_width": 1.96 * float(accuracy_std) / math.sqrt(len(accuracies)) if accuracies else "",
                      "absolute_calibration_test_gap_mean": statistics.fmean(abs(value) for value in gaps) if gaps else "",
                      "heldout_clipped_mode_count_mean": statistics.fmean(clipped) if clipped else "",
                      "runs": len(group)})
    mean_fields = ["dataset", "channel", "level", "dependence_score_mean", "dependence_score_std",
                   "dependence_score_ci95_half_width", "probe_accuracy_mean", "probe_accuracy_std",
                   "probe_accuracy_ci95_half_width", "absolute_calibration_test_gap_mean",
                   "heldout_clipped_mode_count_mean", "runs"]
    temporary = output / "tsd_utility_means.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=mean_fields, lineterminator="\n"); writer.writeheader(); writer.writerows(means)
    temporary.replace(output / "tsd_utility_means.csv")
    calibration_rows = []; chain_rows = []
    for calibration_path in sorted(Path("runs").glob("*/artifacts/e7_tsd_calibration*.json")):
        status_path = calibration_path.parents[1] / "status.json"
        if not status_path.is_file() or json.loads(status_path.read_text(encoding="utf-8")).get("state") != "SUCCEEDED":
            continue
        payload = read_json(calibration_path)
        if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            continue
        feature_dim = dict(payload.get("parameters", {})).get("feature_dim", "")
        for condition, values in sorted(dict(payload.get("calibration_summary", {})).items()):
            calibration_rows.append({"run_id": calibration_path.parents[1].name, "feature_dim": feature_dim,
                                     "condition": condition, **dict(values)})
        for value in payload.get("data_processing_chain", []):
            chain_rows.append({"run_id": calibration_path.parents[1].name, "feature_dim": feature_dim, **dict(value)})
    calibration_fields = sorted({key for row in calibration_rows for key in row}) if calibration_rows else ["run_id"]
    temporary = output / "tsd_calibration_summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=calibration_fields, lineterminator="\n"); writer.writeheader(); writer.writerows(calibration_rows)
    temporary.replace(output / "tsd_calibration_summary.csv")
    chain_fields = sorted({key for row in chain_rows for key in row}) if chain_rows else ["run_id"]
    temporary = output / "tsd_data_processing_chain.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=chain_fields, lineterminator="\n"); writer.writeheader(); writer.writerows(chain_rows)
    temporary.replace(output / "tsd_data_processing_chain.csv")
    image_chain_rows = []
    for manifest in sorted(Path("runs").glob("*/artifacts/image_chain_submitted.json")):
        for record in json.loads(manifest.read_text(encoding="utf-8")):
            source_run = str(record["run_id"]); probe_run = str(record.get("probe_run", ""))
            if not has_postfix_train_result(source_run):
                continue
            evaluation_path = Path("runs") / source_run / "artifacts" / "evaluation.json"
            probe_path = Path("runs") / probe_run / "artifacts" / "probe_result.json"
            heldout = ""; accuracy = ""; clipped = ""
            if evaluation_path.is_file():
                evaluation = read_json(evaluation_path)
                eigenvalues = evaluation.get("test_empirical_eigenvalues", []) if evaluation.get("scientific_correctness_version") == SCIENTIFIC_CORRECTNESS_VERSION else []
                if eigenvalues:
                    clipped = sum(float(value) >= 1.0 - 1e-7 for value in eigenvalues)
                    heldout = sum(-math.log1p(-min(max(float(value), 0.0), 1.0 - 1e-7)) for value in eigenvalues)
            if probe_path.is_file(): accuracy = json.loads(probe_path.read_text(encoding="utf-8")).get("test_accuracy", "")
            image_chain_rows.append({"stage": record["stage"], "seed_index": record["seed_index"], "seed": record["seed"],
                                     "source_run": source_run, "probe_run": probe_run, "heldout_tsd": heldout,
                                     "heldout_clipped_mode_count": clipped, "probe_accuracy": accuracy,
                                     "augmentation_json": json.dumps(record["augmentation"], sort_keys=True, separators=(",", ":"))})
    image_chain_fields = ["stage", "seed_index", "seed", "source_run", "probe_run", "heldout_tsd", "heldout_clipped_mode_count", "probe_accuracy", "augmentation_json"]
    temporary = output / "image_data_processing_chain.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=image_chain_fields, lineterminator="\n"); writer.writeheader(); writer.writerows(image_chain_rows)
    temporary.replace(output / "image_data_processing_chain.csv")
    image_chain_summary = []
    for stage in sorted({int(row["stage"]) for row in image_chain_rows}):
        selected = [row for row in image_chain_rows if int(row["stage"]) == stage]
        tsd_values = [float(row["heldout_tsd"]) for row in selected if row["heldout_tsd"] != ""]
        utility_values = [float(row["probe_accuracy"]) for row in selected if row["probe_accuracy"] != ""]
        image_chain_summary.append({"stage": stage, "runs": len(selected),
                                    "heldout_tsd_mean": statistics.fmean(tsd_values) if tsd_values else "",
                                    "probe_accuracy_mean": statistics.fmean(utility_values) if utility_values else ""})
    plotted_chain = [
        row for row in image_chain_summary
        if row["heldout_tsd_mean"] != "" and row["probe_accuracy_mean"] != ""
    ]
    if plotted_chain:
        width, height, left, top, plot_w, plot_h = 900, 430, 75, 55, 750, 300
        tsd_values = [float(row["heldout_tsd_mean"]) for row in plotted_chain]
        utility_values = [float(row["probe_accuracy_mean"]) for row in plotted_chain]
        def normalized(value: float, values: list[float]) -> float:
            span = max(values) - min(values); return 0.5 if span == 0 else (value - min(values)) / span
        points_tsd = []; points_utility = []
        for index, row in enumerate(plotted_chain):
            x = left + plot_w * index / max(1, len(plotted_chain) - 1)
            points_tsd.append(f"{x:.2f},{top + plot_h * (1-normalized(float(row['heldout_tsd_mean']), tsd_values)):.2f}")
            points_utility.append(f"{x:.2f},{top + plot_h * (1-normalized(float(row['probe_accuracy_mean']), utility_values)):.2f}")
        svg_chain = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                     '<style>.axis{stroke:#222}.tsd{fill:none;stroke:#2855a6;stroke-width:3}.utility{fill:none;stroke:#d14b3f;stroke-width:3}.label{font:12px sans-serif}.title{font:600 16px sans-serif}</style>',
                     '<text x="20" y="28" class="title">Cumulative crop → color → blur: held-out TSD and utility</text>',
                     f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/><line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>',
                     f'<polyline points="{" ".join(points_tsd)}" class="tsd"/><polyline points="{" ".join(points_utility)}" class="utility"/>',
                     '<text x="610" y="28" class="label" fill="#2855a6">held-out TSD (normalized)</text><text x="750" y="28" class="label" fill="#d14b3f">utility (normalized)</text>']
        for index, row in enumerate(plotted_chain):
            x = left + plot_w * index / max(1, len(plotted_chain) - 1); svg_chain.append(f'<text x="{x:.2f}" y="{top+plot_h+20}" text-anchor="middle" class="label">{row["stage"]}</text>')
        svg_chain.append('</svg>'); atomic_text(output / "image_data_processing_chain.svg", "".join(svg_chain))
    plotted = [row for row in means if row["probe_accuracy_mean"] != ""]
    width = 1200; height = 720; margin = 70; plot_w = 1050; plot_h = 570
    colors = {"cifar10": "#2855a6", "cifar100": "#d14b3f", "imagenet100": "#228b22"}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<style>.label{font:12px sans-serif}.title{font:600 16px sans-serif}.axis{stroke:#222}.grid{stroke:#ddd}</style>',
           '<text x="20" y="28" class="title">E7 TSD severity versus frozen linear-probe utility</text>']
    if plotted:
        xs = [float(row["dependence_score_mean"]) for row in plotted]; ys = [float(row["probe_accuracy_mean"]) for row in plotted]
        x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys); xspan = max(x1-x0, 1e-9); yspan = max(y1-y0, 1e-9)
        svg.append(f'<line x1="{margin}" y1="{margin+plot_h}" x2="{margin+plot_w}" y2="{margin+plot_h}" class="axis"/><line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin+plot_h}" class="axis"/>')
        for row in plotted:
            x = margin + plot_w * (float(row["dependence_score_mean"])-x0)/xspan; y = margin + plot_h * (1-(float(row["probe_accuracy_mean"])-y0)/yspan)
            color = colors.get(str(row["dataset"]), "#555"); svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/><title>{row["dataset"]} {row["channel"]} level {row["level"]}</title>')
        svg.append(f'<text x="{margin+plot_w/2-80}" y="{height-20}" class="label">held-out dependence score</text><text x="10" y="{margin+plot_h/2}" class="label" transform="rotate(-90 10 {margin+plot_h/2})">linear-probe accuracy</text>')
    svg.append("</svg>"); atomic_text(output / "tsd_utility.svg", "".join(svg))
    atomic_text(output / "tsd_utility_caption.txt",
                "Post-fix E7 augmentation-severity held-out TSD and frozen linear-probe utility. Only versioned full-matrix held-out spectra are included; clipping counts and calibration-test gaps are explicit, and group tables include seed std/95% CI. The analytic calibration table reports slope, R-squared, Spearman, absolute error, split-half test-retest reliability, monotonicity violations, and calibration-test gap; the chain table retains each cumulative data-processing stage.\n")
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "render_e7_tsd_assets", "sources": len(rows), "paired": len(plotted)}) + "\n")
    print(json.dumps({"sources": len(rows), "paired_groups": len(plotted), "output": str(output)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
