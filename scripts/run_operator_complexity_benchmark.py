#!/usr/bin/env python3
"""E10 isolated moment/whitening/SVD CUDA complexity benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from fmca_av.operators import estimate_moments, fit_spectral_calibration


def timed(operation, warmup: int, iterations: int) -> list[float]:
    values = []
    for index in range(warmup + iterations):
        torch.cuda.synchronize(); started = time.perf_counter(); operation(); torch.cuda.synchronize()
        if index >= warmup: values.append(time.perf_counter() - started)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="")
    parser.add_argument("--batch", type=int, default=512); parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3); parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20305000); args = parser.parse_args()
    torch.manual_seed(args.seed); device = torch.device("cuda"); records = []
    for features in (32, 64, 128, 256, 512):
        f = torch.randn(args.batch, features, device=device)
        g = torch.randn(args.batch, args.views, features, device=device)
        torch.cuda.reset_peak_memory_stats()
        moment_times = timed(lambda: estimate_moments(f, g, centered=True), args.warmup, args.iterations)
        calibration_times = timed(
            lambda: fit_spectral_calibration(f, g, ridge=1e-3, centered=True),
            args.warmup, args.iterations,
        )
        records.append({
            "features": features, "batch": args.batch, "views": args.views,
            "moment_seconds_median": statistics.median(moment_times),
            "moment_seconds_mean": statistics.fmean(moment_times),
            "calibration_seconds_median": statistics.median(calibration_times),
            "calibration_seconds_mean": statistics.fmean(calibration_times),
            "calibration_examples_per_second": args.batch * args.views / statistics.fmean(calibration_times),
            "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
            "iterations": args.iterations, "status": "success",
        })
        del f, g; torch.cuda.empty_cache()
    payload = {"device": torch.cuda.get_device_name(), "conditions": records, "parameters": vars(args)}
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "operator_complexity.json"
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "operator_complexity", "conditions": len(records)}) + "\n")
    print(json.dumps(payload, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
