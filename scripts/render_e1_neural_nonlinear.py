#!/usr/bin/env python3
"""Merge and render the completed post-fix neural nonlinear E1 shards."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import statistics

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


FAMILIES = ("two_moons", "gmm", "spiral")
METRICS = (
    "topk_spectrum_mae", "topk_spectrum_relative_l1", "heldout_trace",
    "oracle_topk_trace", "final_validation_score", "training_duration_seconds",
)


def read_inputs(paths: list[str]) -> list[dict[str, object]]:
    records = []
    for value in paths:
        path = Path(value).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
            raise RuntimeError(f"refusing pre-fix neural nonlinear input: {path}")
        if payload.get("method") != "fmca_av_lightning_mlp":
            raise RuntimeError(f"unexpected method in {path}")
        for raw in payload.get("records", []):
            record = dict(raw); record["source"] = str(path); records.append(record)
    keys = [(str(record["family"]), int(record["seed_index"])) for record in records]
    expected = {(family, seed) for family in FAMILIES for seed in (1, 2, 3)}
    if set(keys) != expected or len(keys) != len(expected):
        raise RuntimeError(f"expected exactly nine unique family/seed records, got {keys}")
    if any(record.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION
           for record in records):
        raise RuntimeError("record-level scientific version mismatch")
    return sorted(records, key=lambda record: (FAMILIES.index(str(record["family"])), int(record["seed_index"])))


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8"); temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    temporary.replace(path)


def svg(summary: list[dict[str, object]]) -> str:
    width, height = 780, 430; left, top, plot_w, plot_h = 85, 55, 650, 285
    maximum = max(float(row["topk_spectrum_mae_mean"]) + float(row["topk_spectrum_mae_std"]) for row in summary)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>.axis{stroke:#222;stroke-width:1.5}.bar{fill:#2855a6}.label{font:13px sans-serif}.title{font:600 16px sans-serif}.note{font:12px sans-serif;fill:#444}</style>',
             '<text x="390" y="25" text-anchor="middle" class="title">Neural FMCA-AV nonlinear held-out spectrum error</text>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/><line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>']
    for step in range(5):
        value = maximum * step / 4; y = top + plot_h * (1.0 - step / 4)
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="note">{value:.3f}</text>')
    slot = plot_w / len(summary)
    for index, row in enumerate(summary):
        mean = float(row["topk_spectrum_mae_mean"]); std = float(row["topk_spectrum_mae_std"])
        x = left + slot * index + slot * 0.24; bar_w = slot * 0.52
        y = top + plot_h * (1.0 - mean / maximum); bar_h = top + plot_h - y
        center = x + bar_w / 2; err_top = top + plot_h * (1.0 - min(maximum, mean + std) / maximum)
        err_bottom = top + plot_h * (1.0 - max(0.0, mean - std) / maximum)
        parts += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" class="bar"/>',
                  f'<line x1="{center:.1f}" y1="{err_top:.1f}" x2="{center:.1f}" y2="{err_bottom:.1f}" stroke="#111"/>',
                  f'<text x="{center:.1f}" y="{top + plot_h + 22}" text-anchor="middle" class="label">{escape(str(row["family"]))}</text>',
                  f'<text x="{center:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="note">{mean:.3f}</text>']
    parts.append('<text x="390" y="395" text-anchor="middle" class="note">Bars: mean over 3 seeds; whiskers: sample standard deviation. Lower is better.</text></svg>')
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    records = read_inputs(args.input)
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    raw_fields = [
        "family", "seed_index", "seed", "views", "noise_std", "max_epochs",
        "global_optimizer_step", "train_samples", "calibration_samples", "test_samples",
        "oracle_samples", "trainable_parameters", *METRICS, "checkpoint", "source",
    ]
    write_csv(output / "neural_nonlinear_runs.csv", records, raw_fields)
    summary = []
    for family in FAMILIES:
        values = [record for record in records if record["family"] == family]
        row: dict[str, object] = {"family": family, "seeds": len(values)}
        for metric in METRICS:
            samples = [float(record[metric]) for record in values]
            row[metric + "_mean"] = statistics.fmean(samples)
            row[metric + "_std"] = statistics.stdev(samples)
        summary.append(row)
    summary_fields = ["family", "seeds", *[suffix for metric in METRICS for suffix in (metric + "_mean", metric + "_std")]]
    write_csv(output / "neural_nonlinear_summary.csv", summary, summary_fields)
    combined = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "method": "fmca_av_lightning_mlp", "records": records, "summary": summary,
    }
    atomic_text(output / "neural_nonlinear_results.json", json.dumps(combined, indent=2, sort_keys=True) + "\n")
    atomic_text(output / "neural_nonlinear_spectrum_error.svg", svg(summary))
    caption = (
        "Post-fix Lightning FMCA-AV MLP pilot on continuous two moons, GMM, and spiral parents. "
        "Each family uses three seeds, M=8 independent additive-Gaussian subviews, 100 epochs, "
        "held-out full-matrix canonical SVD, and a fixed-bin 100k-parent numerical oracle. "
        "The plotted top-k spectrum MAE is a numerical-oracle approximation error; these pilot "
        "results are retained whether favorable or unfavorable and do not replace the exact finite-channel evidence.\n"
    )
    atomic_text(output / "neural_nonlinear_caption.txt", caption)
    print(json.dumps({"records": len(records), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
