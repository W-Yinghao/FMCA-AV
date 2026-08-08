#!/usr/bin/env python3
"""Render E0/E1 exact/operator-recovery tables and vector figure."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from html import escape
import json
import math
from pathlib import Path
import statistics


PALETTE = ("#2855a6", "#d14b3f", "#20854e", "#7a4aa8", "#d18b16", "#327b8e", "#854b3f")


def line_panel(x: float, y: float, width: float, height: float, title: str, x_label: str, y_label: str, series: dict[str, list[tuple[float, float]]]) -> str:
    left, top, right, bottom = 67, 31, 18, 48; px, py = x + left, y + top; pw, ph = width - left - right, height - top - bottom
    all_x = [point[0] for values in series.values() for point in values]; all_y = [max(point[1], 1e-12) for values in series.values() for point in values]
    x0, x1 = math.log10(min(all_x)), math.log10(max(all_x)); y0, y1 = math.log10(min(all_y)), math.log10(max(all_y))
    if y1 == y0: y1 = y0 + 1
    sx = lambda value: px + pw * (math.log10(value) - x0) / (x1 - x0)
    sy = lambda value: py + ph * (y1 - math.log10(max(value, 1e-12))) / (y1 - y0)
    parts = [f'<g><text x="{x + width / 2}" y="{y + 19}" text-anchor="middle" class="title">{escape(title)}</text>', f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" class="axis"/><line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" class="axis"/>']
    for step in range(5):
        fraction = step / 4; yy = py + ph * fraction; value = 10 ** (y1 - fraction * (y1 - y0)); parts += [f'<line x1="{px}" y1="{yy}" x2="{px + pw}" y2="{yy}" class="grid"/>', f'<text x="{px - 6}" y="{yy + 4}" text-anchor="end" class="tick">{value:.2g}</text>']
    for index, (label, values) in enumerate(sorted(series.items())):
        color = PALETTE[index % len(PALETTE)]; points = " ".join(f"{sx(xv):.1f},{sy(yv):.1f}" for xv, yv in values); parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.1"/>')
        for xv, yv in values: parts.append(f'<circle cx="{sx(xv):.1f}" cy="{sy(yv):.1f}" r="3.4" fill="{color}"><title>{escape(label)}: x={xv:g}, y={yv:.4g}</title></circle>')
    for value in sorted(set(all_x)): parts.append(f'<text x="{sx(value):.1f}" y="{py + ph + 17}" text-anchor="middle" class="tick">{value:g}</text>')
    parts += [f'<text x="{x + width / 2}" y="{y + height - 6}" text-anchor="middle" class="label">{escape(x_label)} (log)</text>', f'<text x="{x + 13}" y="{y + height / 2}" text-anchor="middle" transform="rotate(-90 {x + 13} {y + height / 2})" class="label">{escape(y_label)} (log)</text>', '</g>']; return "".join(parts)


def bar_panel(x: float, y: float, width: float, height: float, values: list[tuple[str, float]]) -> str:
    px, py, pw, ph = x + 66, y + 31, width - 84, height - 82; maximum = max(value for _, value in values); parts = [f'<g><text x="{x + width / 2}" y="{y + 19}" text-anchor="middle" class="title">Exact discrete-channel dependence</text>', f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" class="axis"/><line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" class="axis"/>']
    slot = pw / len(values)
    for index, (label, value) in enumerate(values):
        bar_height = ph * value / maximum; bx = px + slot * index + slot * 0.18; bw = slot * 0.64; parts.append(f'<rect x="{bx:.1f}" y="{py + ph - bar_height:.1f}" width="{bw:.1f}" height="{bar_height:.1f}" fill="{PALETTE[index % len(PALETTE)]}"><title>{escape(label)} median trace={value:.4g}</title></rect>'); parts.append(f'<text x="{bx + bw / 2:.1f}" y="{py + ph + 12}" text-anchor="end" transform="rotate(-35 {bx + bw / 2:.1f} {py + ph + 12})" class="tick">{escape(label)}</text>')
    parts += [f'<text x="{x + 13}" y="{y + height / 2}" text-anchor="middle" transform="rotate(-90 {x + 13} {y + height / 2})" class="label">median trace dependence</text>', '</g>']; return "".join(parts)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else ["empty"]; temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--gaussian", required=True); parser.add_argument("--gaussian-extra", action="append", default=[]); parser.add_argument("--nonlinear", required=True); parser.add_argument("--discrete", required=True); parser.add_argument("--finite-sample", default=""); parser.add_argument("--output-dir", default="results/e1")
    args = parser.parse_args(); output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    gaussian = json.loads(Path(args.gaussian).read_text(encoding="utf-8"))["records"]; nonlinear = json.loads(Path(args.nonlinear).read_text(encoding="utf-8"))["records"]; discrete = json.loads(Path(args.discrete).read_text(encoding="utf-8"))["records"]
    for extra in args.gaussian_extra:
        gaussian.extend(json.loads(Path(extra).read_text(encoding="utf-8"))["records"])
    grouped = defaultdict(list)
    for record in gaussian: grouped[(int(record["dimension"]), int(record["samples"]))].append(record)
    gaussian_rows = []
    for (dimension, samples), records in sorted(grouped.items()): gaussian_rows.append({"dimension": dimension, "samples": samples, "spectrum_mae_median": statistics.median(float(record["spectrum_mae"]) for record in records), "top_projector_error_median": statistics.median(float(record["top_projector_error"]) for record in records), "conditions": len(records)})
    grouped = defaultdict(list)
    for record in nonlinear: grouped[(str(record["family"]), int(record["samples"]))].append(record)
    nonlinear_rows = [{
        "family": family, "samples": samples,
        "top16_spectrum_mae_median": statistics.median(float(value["top16_spectrum_mae"]) for value in values),
        "top_subspace_projector_error_median": statistics.median(float(value["top_subspace_projector_error"]) for value in values) if values and "top_subspace_projector_error" in values[0] else "",
        "density_ratio_weighted_rmse_median": statistics.median(float(value["density_ratio_weighted_rmse"]) for value in values) if values and "density_ratio_weighted_rmse" in values[0] else "",
        "density_ratio_excess_rmse_median": statistics.median(float(value["density_ratio_excess_rmse"]) for value in values) if values and "density_ratio_excess_rmse" in values[0] else "",
        "conditions": len(values),
    } for (family, samples), values in sorted(grouped.items())]
    grouped = defaultdict(list)
    for record in discrete: grouped[str(record["family"])].append(float(record["trace_dependence"]))
    discrete_rows = [{"family": family, "trace_dependence_median": statistics.median(values), "conditions": len(values)} for family, values in sorted(grouped.items())]
    write_csv(output / "gaussian_recovery.csv", gaussian_rows); write_csv(output / "nonlinear_recovery.csv", nonlinear_rows); write_csv(output / "exact_discrete_channels.csv", discrete_rows)
    finite_rows = []
    if args.finite_sample:
        finite_records = json.loads(Path(args.finite_sample).read_text(encoding="utf-8"))["records"]
        finite_grouped = defaultdict(list)
        for record in finite_records: finite_grouped[(str(record["family"]), int(record["samples"]))].append(record)
        for (family, samples), records in sorted(finite_grouped.items()):
            finite_rows.append({
                "family": family, "samples": samples, "replicates": len(records),
                "spectrum_mae_median": statistics.median(float(record["spectrum_mae"]) for record in records),
                "spectrum_relative_l1_median": statistics.median(float(record["spectrum_relative_l1"]) for record in records),
                "top_projector_error_median": statistics.median(float(record["top_projector_error"]) for record in records),
                "density_ratio_weighted_rmse_median": statistics.median(float(record["density_ratio_weighted_rmse"]) for record in records),
                "oracle_truncation_density_ratio_weighted_rmse_median": statistics.median(float(record["oracle_truncation_density_ratio_weighted_rmse"]) for record in records),
                "density_ratio_excess_rmse_median": statistics.median(float(record["density_ratio_excess_rmse"]) for record in records),
            })
        write_csv(output / "finite_sample_recovery.csv", finite_rows)
    spectrum = defaultdict(list); projector = defaultdict(list); nonlinear_series = defaultdict(list)
    for row in gaussian_rows: spectrum[f"d={row['dimension']}"].append((float(row["samples"]), float(row["spectrum_mae_median"]))); projector[f"d={row['dimension']}"].append((float(row["samples"]), float(row["top_projector_error_median"])))
    for row in nonlinear_rows: nonlinear_series[str(row["family"])].append((float(row["samples"]), float(row["top16_spectrum_mae_median"])))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="760" viewBox="0 0 1100 760"><style>.axis{stroke:#222;stroke-width:1.4}.grid{stroke:#ddd}.title{font:600 15px sans-serif}.label{font:13px sans-serif}.tick{font:10px sans-serif;fill:#333}</style>', line_panel(10, 10, 535, 360, "Gaussian spectrum recovery", "samples", "median spectrum MAE", spectrum), line_panel(555, 10, 535, 360, "Gaussian subspace recovery", "samples", "median projector error", projector), line_panel(10, 385, 535, 360, "Nonlinear toy recovery", "samples", "median top-16 spectrum MAE", nonlinear_series), bar_panel(555, 385, 535, 360, [(row["family"], float(row["trace_dependence_median"])) for row in discrete_rows]), '</svg>']
    temporary = output / "exact_operator_recovery.svg.tmp"; temporary.write_text("".join(svg), encoding="utf-8"); temporary.replace(output / "exact_operator_recovery.svg")
    caption = "E0/E1 exact and learned operator recovery. Curves aggregate all preregistered noise/view/case replicates by median at each sample size, including the predeclared high-dimensional N=20,000 diagnostic extension; the discrete panel reports exact nonconstant dependence spectra summarized by family. finite_sample_recovery.csv adds 20-repeat empirical spectrum, subspace, and density-ratio errors at three sample sizes for the four decisive finite-channel families. Claim IDs: E0/E1/C1. Sources: " + ", ".join(value for value in (args.gaussian, *args.gaussian_extra, args.nonlinear, args.discrete, args.finite_sample) if value) + "\n"
    temporary = output / "exact_operator_recovery_caption.txt.tmp"; temporary.write_text(caption, encoding="utf-8"); temporary.replace(output / "exact_operator_recovery_caption.txt")
    print(json.dumps({"gaussian_rows": len(gaussian_rows), "nonlinear_rows": len(nonlinear_rows), "discrete_rows": len(discrete_rows), "output_dir": str(output)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
