#!/usr/bin/env python3
"""Aggregate validation-tuned E1 estimator controls into CSV/SVG assets."""

from __future__ import annotations

from collections import defaultdict
import csv
from html import escape
import json
from pathlib import Path
import statistics


def main() -> int:
    candidates = sorted(Path("runs").glob("*/artifacts/e1_estimator_baselines.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("no E1 estimator-baseline artifact")
    source = candidates[-1]
    records = json.loads(source.read_text(encoding="utf-8"))["records"]
    grouped: dict[tuple[str, float, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["method"]), float(record["rho"]), int(record["samples"]))].append(record)
    rows = []
    for (method, rho, samples), values in sorted(grouped.items()):
        spectrum = [float(record["spectrum_mae"]) for record in values if record.get("spectrum_mae") is not None]
        test_spectrum = [float(record["test_spectrum_mae"]) for record in values if record.get("test_spectrum_mae") is not None]
        hsic = [float(record["hsic"]) for record in values if record.get("hsic") is not None]
        test_hsic = [float(record["test_hsic"]) for record in values if record.get("test_hsic") is not None]
        rows.append({
            "method": method, "rho": rho, "samples": samples, "replicates": len(values),
            "spectrum_mae_median": statistics.median(spectrum) if spectrum else "",
            "spectrum_mae_q25": statistics.quantiles(spectrum, n=4)[0] if len(spectrum) > 1 else (spectrum[0] if spectrum else ""),
            "spectrum_mae_q75": statistics.quantiles(spectrum, n=4)[2] if len(spectrum) > 1 else (spectrum[0] if spectrum else ""),
            "test_spectrum_mae_median": statistics.median(test_spectrum) if test_spectrum else "",
            "normalized_hsic_median": statistics.median(hsic) if hsic else "",
            "test_normalized_hsic_median": statistics.median(test_hsic) if test_hsic else "",
        })
    output = Path("results/e1"); output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = output / "estimator_baselines.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(output / "estimator_baselines.csv")
    plotted = [row for row in rows if row["spectrum_mae_median"] != "" and float(row["rho"]) == 0.6]
    methods = sorted({str(row["method"]) for row in plotted}); sizes = sorted({int(row["samples"]) for row in plotted})
    width, height = 940, 460; left, top, chart_w, chart_h = 85, 55, 760, 330
    maximum = max(float(row["spectrum_mae_median"]) for row in plotted)
    palette = ("#2855a6", "#d14b3f", "#20854e", "#7a4aa8")
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<style>.t{font:600 16px sans-serif}.l{font:12px sans-serif}.a{stroke:#222}.g{stroke:#ddd}</style>',
           '<text x="20" y="28" class="t">E1 validation-tuned estimator controls (rho=0.6)</text>',
           f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" class="a"/><line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" class="a"/>']
    for method_index, method in enumerate(methods):
        points = []
        for size_index, size in enumerate(sizes):
            row = next(item for item in plotted if item["method"] == method and int(item["samples"]) == size)
            x = left + chart_w * size_index / max(1, len(sizes) - 1)
            y = top + chart_h * float(row["spectrum_mae_median"]) / max(maximum, 1e-12)
            y = top + chart_h - (y - top)
            points.append(f"{x:.1f},{y:.1f}")
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{palette[method_index % len(palette)]}"/>')
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{palette[method_index % len(palette)]}" stroke-width="2"><title>{escape(method)}</title></polyline>')
        svg.append(f'<text x="{left+chart_w+12}" y="{top+18*method_index+14}" class="l" fill="{palette[method_index % len(palette)]}">{escape(method)}</text>')
    for size_index, size in enumerate(sizes):
        x = left + chart_w * size_index / max(1, len(sizes) - 1)
        svg.append(f'<text x="{x:.1f}" y="{top+chart_h+20}" text-anchor="middle" class="l">{size}</text>')
    svg.append('</svg>')
    temporary = output / "estimator_baselines.svg.tmp"; temporary.write_text("".join(svg), encoding="utf-8"); temporary.replace(output / "estimator_baselines.svg")
    caption = ("E1 Gaussian estimator controls: linear CCA, exact Hermite features, validation-tuned RBF Nyström, "
               "random-Fourier KICA approximation, and normalized HSIC. Bandwidth/ridge selection uses an independent "
               "validation split. Primary spectrum errors use the requested-size training split; separate 10,000-sample "
               "test diagnostics are retained in the CSV. Source: " + str(source) + "\n")
    temporary = output / "estimator_baselines_caption.txt.tmp"; temporary.write_text(caption, encoding="utf-8"); temporary.replace(output / "estimator_baselines_caption.txt")
    print(json.dumps({"rows": len(rows), "source": str(source)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
