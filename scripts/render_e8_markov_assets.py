#!/usr/bin/env python3
"""Render E8 direct/composed Markov validation and boundary assets."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from html import escape
import json
import math
from pathlib import Path
import statistics


PALETTE = ("#2855a6", "#d14b3f", "#20854e", "#7a4aa8", "#d18b16")


def panel(x: float, y: float, width: float, height: float, title: str, label: str, series: dict[str, list[tuple[int, float]]]) -> str:
    left, top, right, bottom = 68, 31, 18, 45; px, py = x + left, y + top; pw, ph = width - left - right, height - top - bottom
    lags = sorted({lag for values in series.values() for lag, _ in values}); values = [max(value, 1e-14) for rows in series.values() for _, value in rows]; lo, hi = math.log10(min(values)), math.log10(max(values));
    if hi == lo: hi = lo + 1
    sx = lambda lag: px + pw * lags.index(lag) / max(1, len(lags) - 1); sy = lambda value: py + ph * (hi - math.log10(max(value, 1e-14))) / (hi - lo)
    parts = [f'<g><text x="{x + width / 2}" y="{y + 19}" text-anchor="middle" class="title">{escape(title)}</text>', f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" class="axis"/><line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" class="axis"/>']
    for step in range(5):
        fraction = step / 4; yy = py + ph * fraction; value = 10 ** (hi - fraction * (hi - lo)); parts += [f'<line x1="{px}" y1="{yy}" x2="{px + pw}" y2="{yy}" class="grid"/>', f'<text x="{px - 6}" y="{yy + 4}" text-anchor="end" class="tick">{value:.2g}</text>']
    for index, (name, rows) in enumerate(sorted(series.items())):
        color = PALETTE[index % len(PALETTE)]; points = " ".join(f"{sx(lag):.1f},{sy(value):.1f}" for lag, value in rows); parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for lag, value in rows: parts.append(f'<circle cx="{sx(lag):.1f}" cy="{sy(value):.1f}" r="3.5" fill="{color}"><title>{escape(name)} lag={lag}, value={value:.4g}</title></circle>')
    for lag in lags: parts.append(f'<text x="{sx(lag):.1f}" y="{py + ph + 17}" text-anchor="middle" class="tick">{lag}</text>')
    parts += [f'<text x="{x + width / 2}" y="{y + height - 5}" text-anchor="middle" class="label">lag</text>', f'<text x="{x + 13}" y="{y + height / 2}" text-anchor="middle" transform="rotate(-90 {x + 13} {y + height / 2})" class="label">{escape(label)} (log)</text>', '</g>']; return "".join(parts)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else ["empty"]; temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--output-dir", default="results/e8")
    args = parser.parse_args(); source = Path(args.input).resolve(); output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True); payload = json.loads(source.read_text(encoding="utf-8"))
    exact_values = defaultdict(list)
    exact_condition_rows = []
    for record in payload["exact_records"]:
        for lag in record["lags"]:
            exact_values[(record["chain"], int(lag["lag"]))].append(float(lag["mae"]))
            exact_condition_rows.append({
                "states": record["states"], "replicate": record["replicate"], "chain": record["chain"],
                "lag": lag["lag"], **{key: value for key, value in lag.items() if key not in {"lag", "direct", "composed", "power"}},
            })
    exact_rows = []
    for (chain, lag), values in sorted(exact_values.items()):
        standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        exact_rows.append({"chain": chain, "lag": lag,
                           "power_spectrum_mae_median": statistics.median(values),
                           "power_spectrum_mae_mean": statistics.fmean(values),
                           "power_spectrum_mae_std": standard_deviation,
                           "power_spectrum_mae_ci95_half_width": 1.96 * standard_deviation / math.sqrt(len(values)),
                           "replicates": len(values)})
    continuous_values = defaultdict(lambda: defaultdict(list))
    continuous_condition_rows = []
    for record in payload["continuous_records"]:
        for lag in record["lags"]:
            key = (record["dynamics"], int(lag["lag"])); continuous_values[key]["spectrum_mae"].append(float(lag["spectrum_mae"])); continuous_values[key]["chapman_kolmogorov_max_abs"].append(float(lag["chapman_kolmogorov_max_abs"])); continuous_values[key]["transition_weighted_l2"].append(float(lag["transition_weighted_l2"]))
            condition = dict(record["condition"]); axis = str(condition.get("axis", "reference"))
            axis_key = {"trajectory_length": "length", "sampling_step": "step", "discretization": "bins",
                        "diffusion_noise": "diffusion", "observation_interval": "observation_stride",
                        "initial_distribution": "initial"}.get(axis, "")
            continuous_condition_rows.append({
                "dynamics": record["dynamics"], "replicate": record["replicate"], "axis": axis,
                "axis_value": condition.get(axis_key, ""), "lag": lag["lag"],
                "trajectory_length": record.get("trajectory_length", condition.get("length", "")),
                "bins": record.get("bins", condition.get("bins", "")),
                "step": condition.get("step", ""), "diffusion": condition.get("diffusion", ""),
                "observation_stride": condition.get("observation_stride", 1), "initial": condition.get("initial", ""),
                "spectrum_mae": lag["spectrum_mae"], "chapman_kolmogorov_max_abs": lag["chapman_kolmogorov_max_abs"],
                "transition_weighted_l2": lag["transition_weighted_l2"],
            })
    continuous_rows = []
    for (dynamics, lag), values in sorted(continuous_values.items()):
        row = {"dynamics": dynamics, "lag": lag, "conditions": len(values["spectrum_mae"])}
        for metric in ("spectrum_mae", "chapman_kolmogorov_max_abs", "transition_weighted_l2"):
            metric_values = values[metric]
            standard_deviation = statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
            row[metric + "_median"] = statistics.median(metric_values)
            row[metric + "_mean"] = statistics.fmean(metric_values)
            row[metric + "_std"] = standard_deviation
            row[metric + "_ci95_half_width"] = 1.96 * standard_deviation / math.sqrt(len(metric_values))
        continuous_rows.append(row)
    write_csv(output / "exact_markov.csv", exact_rows); write_csv(output / "continuous_markov.csv", continuous_rows)
    write_csv(output / "exact_markov_conditions.csv", exact_condition_rows)
    write_csv(output / "continuous_markov_conditions.csv", continuous_condition_rows)
    exact_series = defaultdict(list); spectrum = defaultdict(list); ck = defaultdict(list); transition = defaultdict(list)
    for row in exact_rows: exact_series[row["chain"]].append((int(row["lag"]), float(row["power_spectrum_mae_median"])))
    for row in continuous_rows:
        spectrum[row["dynamics"]].append((int(row["lag"]), float(row["spectrum_mae_median"]))); ck[row["dynamics"]].append((int(row["lag"]), float(row["chapman_kolmogorov_max_abs_median"]))); transition[row["dynamics"]].append((int(row["lag"]), float(row["transition_weighted_l2_median"])))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="760" viewBox="0 0 1100 760"><style>.axis{stroke:#222;stroke-width:1.4}.grid{stroke:#ddd}.title{font:600 15px sans-serif}.label{font:13px sans-serif}.tick{font:10px sans-serif;fill:#333}</style>', panel(10, 10, 535, 360, "Exact one-step power law", "spectrum MAE", exact_series), panel(555, 10, 535, 360, "Continuous direct vs composed", "spectrum MAE", spectrum), panel(10, 385, 535, 360, "Chapman-Kolmogorov residual", "max absolute residual", ck), panel(555, 385, 535, 360, "Transition reconstruction", "weighted L2", transition), '</svg>']
    temporary = output / "markov_direct_composed.svg.tmp"; temporary.write_text("".join(svg), encoding="utf-8"); temporary.replace(output / "markov_direct_composed.svg")
    caption = "E8 direct-versus-composed Markov validation across reversible/non-reversible exact chains and OU/double-/multi-well continuous dynamics. Curves report medians across preregistered replicates and boundary conditions; aggregate CSVs also report means, standard deviations and 95% CI half-widths. The condition tables preserve every trajectory-length, sampling-step, diffusion, observation-interval, discretization and non-equilibrium cell, including negative composition results. Claim IDs: E8/C6/Case3. Source: " + str(source) + "\n"
    temporary = output / "markov_caption.txt.tmp"; temporary.write_text(caption, encoding="utf-8"); temporary.replace(output / "markov_caption.txt")
    print(json.dumps({"exact_rows": len(exact_rows), "continuous_rows": len(continuous_rows), "output_dir": str(output)}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
