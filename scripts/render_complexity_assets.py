#!/usr/bin/env python3
"""Render the E10 complexity table and vector scaling figure."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
import re

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


DEFAULT_RESULTS_ROOT = Path(
    f"results/postfix/{SCIENTIFIC_CORRECTNESS_VERSION}"
)


def read_versioned_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("scientific_correctness_version")
    if recorded != SCIENTIFIC_CORRECTNESS_VERSION:
        raise ValueError(
            f"refusing pre-fix or mismatched E10 input {path}: "
            f"{recorded!r} != {SCIENTIFIC_CORRECTNESS_VERSION!r}"
        )
    return payload


def coordinates(values: list[float], start: float, extent: float, logarithmic: bool = True) -> list[float]:
    transformed = [math.log10(value) if logarithmic else value for value in values]
    low, high = min(transformed), max(transformed)
    if high == low:
        return [start + extent / 2 for _ in values]
    return [start + extent * (value - low) / (high - low) for value in transformed]


def panel(x: float, y: float, width: float, height: float, title: str, x_label: str, values: list[dict], x_key: str, y_key: str) -> str:
    margin_left, margin_bottom, margin_top, margin_right = 56, 44, 30, 16
    plot_x, plot_y = x + margin_left, y + margin_top
    plot_w, plot_h = width - margin_left - margin_right, height - margin_top - margin_bottom
    x_values = [float(item[x_key]) for item in values]; y_values = [float(item[y_key]) for item in values]
    xs = coordinates(x_values, plot_x, plot_w, True); ys_raw = coordinates(y_values, 0, plot_h, True)
    ys = [plot_y + plot_h - value for value in ys_raw]
    parts = [
        f'<g><text x="{x + width / 2:.1f}" y="{y + 18:.1f}" text-anchor="middle" class="title">{escape(title)}</text>',
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" class="axis"/>',
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" class="axis"/>',
        f'<text x="{x + width / 2:.1f}" y="{y + height - 6:.1f}" text-anchor="middle" class="label">{escape(x_label)} (log scale)</text>',
        f'<text x="{x + 13:.1f}" y="{y + height / 2:.1f}" text-anchor="middle" transform="rotate(-90 {x + 13:.1f} {y + height / 2:.1f})" class="label">{escape(y_key)} (log scale)</text>',
    ]
    for index in range(5):
        fraction = index / 4; grid_y = plot_y + plot_h * fraction
        exponent = math.log10(max(y_values)) - fraction * (math.log10(max(y_values)) - math.log10(min(y_values)))
        parts.append(f'<line x1="{plot_x}" y1="{grid_y:.1f}" x2="{plot_x + plot_w}" y2="{grid_y:.1f}" class="grid"/>')
        parts.append(f'<text x="{plot_x - 7}" y="{grid_y + 4:.1f}" text-anchor="end" class="tick">{10 ** exponent:.3g}</text>')
    points = " ".join(f"{px:.1f},{py:.1f}" for px, py in zip(xs, ys)); parts.append(f'<polyline points="{points}" class="curve"/>')
    for px, py, xv, yv in zip(xs, ys, x_values, y_values):
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" class="point"><title>{x_key}={xv:g}, {y_key}={yv:.5g}</title></circle>')
        parts.append(f'<text x="{px:.1f}" y="{plot_y + plot_h + 17:.1f}" text-anchor="middle" class="tick">{xv:g}</text>')
    parts.append('</g>'); return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--operator", default=""); parser.add_argument("--flops", default=""); parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_ROOT / "e10"))
    args = parser.parse_args(); source = Path(args.input).resolve(); output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    payload = read_versioned_payload(source); records = payload["conditions"]
    fields = sorted({key for record in records for key in record})
    temporary_csv = output / "complexity_table.csv.tmp"
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(records)
    temporary_csv.replace(output / "complexity_table.csv")
    operator_records = []
    if args.operator:
        operator_records = read_versioned_payload(Path(args.operator))["conditions"]
        operator_fields = sorted({key for record in operator_records for key in record})
        temporary = output / "operator_complexity_table.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=operator_fields, lineterminator="\n"); writer.writeheader(); writer.writerows(operator_records)
        temporary.replace(output / "operator_complexity_table.csv")
    flops_records = []
    if args.flops:
        flops_records = read_versioned_payload(Path(args.flops))["conditions"]
        flops_fields = sorted({key for record in flops_records for key in record})
        temporary = output / "flops_profile_table.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=flops_fields, lineterminator="\n"); writer.writeheader(); writer.writerows(flops_records)
        temporary.replace(output / "flops_profile_table.csv")
    ddp_records = []
    jobs_path = Path("harness/state/jobs.json")
    if jobs_path.is_file():
        jobs = json.loads(jobs_path.read_text(encoding="utf-8")).get("jobs", {})
        for run_id, status in jobs.items():
            name = str(status.get("name", "")); match = re.search(r"e10-cifar10-ddp([124])-100step", name)
            if not match or status.get("state") != "SUCCEEDED":
                continue
            result_path = Path("runs") / run_id / "artifacts" / "train_result.json"
            if not result_path.is_file():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
                continue
            gpus = int(match.group(1))
            ddp_records.append({
                "run_id": run_id, "name": name, "gpus": gpus,
                "completed_optimizer_steps": result.get("completed_optimizer_steps", 100),
                "global_parent_batch_size": result.get("global_parent_batch_size", 128),
                "encoded_views": result.get("encoded_views", ""),
                "training_duration_seconds": result.get("training_duration_seconds", ""),
                "encoded_views_per_second": result.get("encoded_views_per_second", ""),
                "peak_memory_mb_per_rank": result.get("peak_memory_mb_per_rank", ""),
                "gpu_hours": result.get("gpu_hours", ""),
            })
    ddp_records.sort(key=lambda record: (int(record["gpus"]), str(record["run_id"])))
    reference_throughput = next((float(record["encoded_views_per_second"]) for record in reversed(ddp_records)
                                 if record["gpus"] == 1 and record["encoded_views_per_second"] != ""), None)
    for record in ddp_records:
        throughput = record["encoded_views_per_second"]
        record["scaling_efficiency_vs_1gpu"] = (
            float(throughput) / (reference_throughput * int(record["gpus"]))
            if throughput != "" and reference_throughput else ""
        )
    ddp_fields = sorted({key for record in ddp_records for key in record}) if ddp_records else ["run_id"]
    temporary = output / "ddp_scaling_table.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ddp_fields, lineterminator="\n"); writer.writeheader(); writer.writerows(ddp_records)
    temporary.replace(output / "ddp_scaling_table.csv")
    successful = [record for record in records if record.get("status") == "success"]
    axes = {
        "views": sorted((record for record in successful if record["axis"] == "views"), key=lambda item: item["views"]),
        "features": sorted((record for record in successful if record["axis"] == "features"), key=lambda item: item["features"]),
        "batch": sorted((record for record in successful if record["axis"] == "batch"), key=lambda item: item["batch"]),
    }
    width, height = 1100, 760
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<style>.axis{stroke:#222;stroke-width:1.4}.grid{stroke:#ddd;stroke-width:1}.curve{fill:none;stroke:#2855a6;stroke-width:2.5}.point{fill:#d14b3f}.title{font:600 15px sans-serif}.label{font:13px sans-serif}.tick{font:11px sans-serif;fill:#333}</style>',
           panel(10, 10, 535, 360, "Throughput scaling with M", "M (views)", axes["views"], "views", "encoded_images_per_second"),
           panel(555, 10, 535, 360, "Memory scaling with M", "M (views)", axes["views"], "views", "peak_memory_mb"),
           panel(10, 385, 535, 360, "Throughput scaling with K", "K (features)", axes["features"], "features", "encoded_images_per_second"),
           panel(555, 385, 535, 360, "Throughput scaling with B", "B (batch)", axes["batch"], "batch", "encoded_images_per_second"), '</svg>']
    temporary_svg = output / "complexity_scaling.svg.tmp"; temporary_svg.write_text("".join(svg), encoding="utf-8"); temporary_svg.replace(output / "complexity_scaling.svg")
    if operator_records:
        operator_svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="390" viewBox="0 0 1100 390"><style>.axis{stroke:#222;stroke-width:1.4}.grid{stroke:#ddd}.curve{fill:none;stroke:#2855a6;stroke-width:2.5}.point{fill:#d14b3f}.title{font:600 15px sans-serif}.label{font:13px sans-serif}.tick{font:11px sans-serif;fill:#333}</style>', panel(10, 10, 535, 360, "Moment construction scaling", "K (features)", operator_records, "features", "moment_seconds_median"), panel(555, 10, 535, 360, "Whitening/SVD calibration scaling", "K (features)", operator_records, "features", "calibration_seconds_median"), '</svg>']
        temporary = output / "operator_complexity_scaling.svg.tmp"; temporary.write_text("".join(operator_svg), encoding="utf-8"); temporary.replace(output / "operator_complexity_scaling.svg")
    latest_ddp = {}
    for record in ddp_records:
        if record["encoded_views_per_second"] != "" and record["gpu_hours"] != "":
            latest_ddp[int(record["gpus"])] = record
    if latest_ddp:
        ddp_values = [latest_ddp[key] for key in sorted(latest_ddp)]
        ddp_svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="390" viewBox="0 0 1100 390"><style>.axis{stroke:#222;stroke-width:1.4}.grid{stroke:#ddd}.curve{fill:none;stroke:#2855a6;stroke-width:2.5}.point{fill:#d14b3f}.title{font:600 15px sans-serif}.label{font:13px sans-serif}.tick{font:11px sans-serif;fill:#333}</style>', panel(10, 10, 535, 360, "Single-node DDP throughput", "GPU count", ddp_values, "gpus", "encoded_views_per_second"), panel(555, 10, 535, 360, "Single-node DDP GPU-hours", "GPU count", ddp_values, "gpus", "gpu_hours"), '</svg>']
        temporary = output / "ddp_scaling.svg.tmp"; temporary.write_text("".join(ddp_svg), encoding="utf-8"); temporary.replace(output / "ddp_scaling.svg")
    caption = (
        "E10 complexity scaling on " + str(payload.get("device", "unknown GPU")) + ". "
        "Each point reports the mean of the timed iterations after warm-up; throughput counts all encoded views, "
        "and memory is peak allocated CUDA memory. Axes M, K, and B denote views per parent, feature dimension, and parent batch size. "
        "The isolated operator table/figure separately time moment construction and complete whitening/SVD for K=32--512. "
        "The DDP table/figure use Lightning-recorded optimizer steps and encoded-view throughput for matched-global-batch 1/2-GPU jobs. "
        "The FLOPs table reports profiler-supported operations for a complete forward/objective/backward step and is explicitly an estimate. "
        "Claim IDs: E10 scaling/efficiency. Sources: " + ", ".join(value for value in (str(source), args.operator, args.flops) if value) + "\n"
    )
    temporary_caption = output / "complexity_caption.txt.tmp"; temporary_caption.write_text(caption, encoding="utf-8"); temporary_caption.replace(output / "complexity_caption.txt")
    print(json.dumps({"records": len(records), "successful": len(successful), "output_dir": str(output)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
