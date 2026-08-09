#!/usr/bin/env python3
"""Render paired equal-encoded-view results from the formal SSL matrix."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


STATE = Path("results/orchestration/matched_compute_state.json")
RUN_PATTERN = re.compile(r"/runs/([^/]+)/artifacts/")
DEFAULT_RESULTS_ROOT = Path(f"results/postfix/{SCIENTIFIC_CORRECTNESS_VERSION}")


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_hours(run_id: str) -> float:
    status = read(Path("runs") / run_id / "status.json")
    if not status.get("start_time") or not status.get("end_time"):
        return 0.0
    seconds = (datetime.fromisoformat(str(status["end_time"])) - datetime.fromisoformat(str(status["start_time"]))).total_seconds()
    return seconds * int(status.get("requested_gpus", 0)) / 3600.0


def probe(run_id: str) -> tuple[float, str]:
    payload = read(Path("runs") / run_id / "artifacts" / "probe_result.json")
    if payload.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing legacy matched-compute probe {run_id}")
    checkpoint = str(payload.get("source_checkpoint") or payload.get("checkpoint") or "")
    match = RUN_PATTERN.search(checkpoint)
    return float(payload["test_accuracy"]), match.group(1) if match else ""


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def svg(rows: list[dict[str, object]]) -> str:
    width = 1400
    height = max(300, 80 + 25 * len(rows))
    left = 600
    scale = 700
    maximum = max((abs(float(row["paired_accuracy_difference_v8_minus_v2"])) for row in rows), default=1.0)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>.t{font:11px sans-serif}.h{font:600 16px sans-serif}.p{fill:#2459a6}.n{fill:#b3472f}.z{stroke:#111}</style>',
             '<text x="20" y="28" class="h">Matched-compute paired accuracy: V=8 minus V=2</text>',
             f'<line x1="{left}" y1="45" x2="{left}" y2="{height-20}" class="z"/>']
    for index, row in enumerate(rows):
        y = 55 + 25 * index
        value = float(row["paired_accuracy_difference_v8_minus_v2"])
        ci = float(row["ci95_half_width"])
        label = f'{row["dataset"]} | {row["method"]} | {row["backbone"] or "default"} | {row["aggregation"] or "default"}'
        length = scale * value / max(maximum, 1e-12) / 2
        x = left if value >= 0 else left + length
        parts.append(f'<text x="20" y="{y+12}" class="t">{label}</text><rect x="{x:.2f}" y="{y}" width="{abs(length):.2f}" height="15" class="{"p" if value >= 0 else "n"}"/><text x="{left+length+5:.2f}" y="{y+12}" class="t">{value:.4f} ± {ci:.4f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", default=str(STATE))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_ROOT / "e5"))
    args = parser.parse_args()
    state_path = Path(args.state_file)
    state = read(state_path)
    if state.get("scientific_correctness_version") != SCIENTIFIC_CORRECTNESS_VERSION:
        raise RuntimeError(f"refusing legacy matched-compute state: {state_path}")
    if str(state.get("state")) != "SUCCEEDED":
        raise RuntimeError("matched-compute state is not SUCCEEDED")
    rows = []
    for pair in list(state["pairs"]):
        record = dict(pair)
        v2_accuracy, v2_source = probe(str(record["v2_probe_run"]))
        v8_accuracy, _ = probe(str(record["v8_probe_run"]))
        v8_source = str(record["v8_source_run"])
        rows.append({
            **record,
            "v2_source_run": v2_source,
            "v2_accuracy": v2_accuracy,
            "v8_accuracy": v8_accuracy,
            "paired_accuracy_difference_v8_minus_v2": v8_accuracy - v2_accuracy,
            "v2_view_epochs": int(record["v2_epochs"]) * 2,
            "v8_view_epochs": int(record["v8_epochs"]) * 8,
            "v2_source_gpu_hours": gpu_hours(v2_source),
            "v8_source_gpu_hours": gpu_hours(v8_source),
        })
    fields = list(rows[0]) if rows else ["key"]
    output = Path(args.output_dir)
    write_csv(output / "matched_compute_runs.csv", rows, fields)
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], row["backbone"], row["aggregation"])].append(float(row["paired_accuracy_difference_v8_minus_v2"]))
    summaries = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        summaries.append({
            "dataset": key[0], "method": key[1], "backbone": key[2], "aggregation": key[3],
            "paired_seeds": len(values), "paired_accuracy_difference_v8_minus_v2": statistics.fmean(values),
            "standard_deviation": standard_deviation,
            "ci95_half_width": 1.96 * standard_deviation / math.sqrt(len(values)),
        })
    summary_fields = list(summaries[0]) if summaries else ["dataset"]
    write_csv(output / "matched_compute_summary.csv", summaries, summary_fields)
    temporary = output / "matched_compute_accuracy.svg.tmp"
    temporary.write_text(svg(summaries), encoding="utf-8")
    temporary.replace(output / "matched_compute_accuracy.svg")
    caption = ("Post-fix equal-encoded-view comparison of V=2 at the full epoch budget and V=8 at one quarter of that budget. "
               "Only state, probes, and source checkpoints carrying the current scientific-correctness version are accepted. "
               "Differences are paired by frozen seed index; GPU-hours are measured from the exact source checkpoint runs. Claim IDs: E5/C3.\n")
    temporary = output / "matched_compute_caption.txt.tmp"
    temporary.write_text(caption, encoding="utf-8")
    temporary.replace(output / "matched_compute_caption.txt")
    print(json.dumps({"pairs": len(rows), "groups": len(summaries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
