#!/usr/bin/env python3
"""Render E2 fixed-parent/fixed-budget bias-variance assets."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path


COLORS = {"fixed_parent": "#2855a6", "fixed_total_views": "#d14b3f"}


def panel(x: float, y: float, width: float, height: float, records: list[dict], metric: str, title: str) -> str:
    left, top, right, bottom = 70, 32, 18, 48; px, py = x + left, y + top; pw, ph = width - left - right, height - top - bottom
    values = [abs(float(record[metric])) for record in records]; low, high = min(value for value in values if value > 0), max(values)
    y0, y1 = math.log10(low), math.log10(high); parts = [f'<g><text x="{x + width / 2}" y="{y + 19}" text-anchor="middle" class="title">{escape(title)}</text>', f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" class="axis"/><line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" class="axis"/>']
    for step in range(5):
        fraction = step / 4; yy = py + ph * fraction; value = 10 ** (y1 - fraction * (y1 - y0)); parts += [f'<line x1="{px}" y1="{yy}" x2="{px + pw}" y2="{yy}" class="grid"/>', f'<text x="{px - 7}" y="{yy + 4}" text-anchor="end" class="tick">{value:.2g}</text>']
    views = sorted({int(record["views"]) for record in records}); xs = {view: px + pw * index / (len(views) - 1) for index, view in enumerate(views)}
    for design in ("fixed_parent", "fixed_total_views"):
        selected = sorted((record for record in records if record["design"] == design), key=lambda record: record["views"]); points = []
        for record in selected:
            xx = xs[int(record["views"])]; value = abs(float(record[metric])); yy = py + ph * (y1 - math.log10(value)) / (y1 - y0); points.append(f"{xx:.1f},{yy:.1f}"); parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{COLORS[design]}"/>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[design]}" stroke-width="2.5"/>')
    for view in views: parts.append(f'<text x="{xs[view]:.1f}" y="{py + ph + 18}" text-anchor="middle" class="tick">{view}</text>')
    parts += [f'<text x="{x + width / 2}" y="{y + height - 7}" text-anchor="middle" class="label">conditional views M</text>', f'<text x="{x + 14}" y="{y + height / 2}" text-anchor="middle" transform="rotate(-90 {x + 14} {y + height / 2})" class="label">{escape(metric)} (log)</text>', '</g>']; return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--output-dir", default="results/e2")
    args = parser.parse_args(); source = Path(args.input).resolve(); output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True); payload = json.loads(source.read_text(encoding="utf-8")); records = payload["conditions"]
    baselines = {str(record["design"]): record for record in records if int(record["views"]) == 1}
    for record in records:
        baseline = baselines[str(record["design"])]
        repetitions = int(record["repetitions"]); baseline_repetitions = int(baseline["repetitions"])
        log_se = math.sqrt(2.0 / max(1, repetitions - 1) + 2.0 / max(1, baseline_repetitions - 1))
        for metric, prefix in (("score_variance", "score_variance_ratio"), ("gradient_variance", "gradient_variance_ratio")):
            denominator = float(baseline[metric])
            ratio = float(record[metric]) / denominator if denominator > 0 else float("nan")
            record[prefix] = ratio
            record[prefix + "_ci95_low"] = ratio * math.exp(-1.96 * log_se)
            record[prefix + "_ci95_high"] = ratio * math.exp(1.96 * log_se)
    fields = sorted({key for record in records for key in record}); temporary = output / "gradient_variance_table.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    temporary.replace(output / "gradient_variance_table.csv")
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="760" viewBox="0 0 1100 760"><style>.axis{stroke:#222;stroke-width:1.4}.grid{stroke:#ddd}.title{font:600 15px sans-serif}.label{font:13px sans-serif}.tick{font:11px sans-serif;fill:#333}</style>', panel(10, 10, 535, 360, records, "score_variance", "Dependence-score variance"), panel(555, 10, 535, 360, records, "gradient_variance", "Gradient variance"), panel(10, 385, 535, 360, records, "score_bias", "Absolute dependence-score bias"), panel(555, 385, 535, 360, records, "gradient_mse_to_reference", "Gradient MSE to reference"), '<g><rect x="785" y="720" width="12" height="3" fill="#2855a6"/><text x="803" y="726" class="tick">fixed parents</text><rect x="890" y="720" width="12" height="3" fill="#d14b3f"/><text x="908" y="726" class="tick">fixed total views</text></g></svg>']
    temporary = output / "gradient_variance.svg.tmp"; temporary.write_text("".join(svg), encoding="utf-8"); temporary.replace(output / "gradient_variance.svg")
    protocol = str(payload.get("parent_protocol", "unspecified parent protocol"))
    caption = "E2 frozen-network conditional-sampling bias/variance over 500 repetitions. Parent protocol: " + protocol + ". Blue holds parent count fixed; red holds total encoded views fixed. All panels use held-out reference estimates and log-scaled magnitude. The CSV includes variance ratios relative to M=1 and log-ratio normal-approximation 95% intervals using the two variance-estimator degrees of freedom. Claim IDs: E2/C2. Source: " + str(source) + "\n"
    temporary = output / "gradient_variance_caption.txt.tmp"; temporary.write_text(caption, encoding="utf-8"); temporary.replace(output / "gradient_variance_caption.txt")
    print(json.dumps({"conditions": len(records), "output_dir": str(output)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
