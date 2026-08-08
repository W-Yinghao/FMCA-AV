#!/usr/bin/env python3
"""E7 held-out Gaussian TSD calibration and data-processing diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from fmca_av.operators import (
    SCIENTIFIC_CORRECTNESS_VERSION,
    clipped_logdet_from_eigenvalues,
    evaluate_heldout_spectrum,
    fit_spectral_calibration,
)


def hermite(values: torch.Tensor, dimension: int) -> torch.Tensor:
    columns = []; previous = torch.ones_like(values); current = values
    for degree in range(1, dimension + 1):
        columns.append(current)
        following = (values * current - math.sqrt(degree) * previous) / math.sqrt(degree + 1)
        previous, current = current, following
    return torch.cat(columns, 1)


def exact_tsd(noise: float, dimension: int) -> float:
    values = torch.tensor([(1.0 / (1.0 + noise)) ** degree for degree in range(1, dimension + 1)], dtype=torch.float64)
    return float(-torch.log1p(-values).sum())


def estimate(noise: float, samples: int, views: int, dimension: int, seed: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(samples, 1, generator=generator, dtype=torch.float64)
    y = x[:, None, :] + math.sqrt(noise) * torch.randn(samples, views, 1, generator=generator, dtype=torch.float64)
    f = hermite(x, dimension); g = hermite(y.reshape(-1, 1) / math.sqrt(1.0 + noise), dimension).reshape(samples, views, dimension)
    calibration = fit_spectral_calibration(f, g, ridge=1e-3, centered=True)
    calibration_tsd_value, calibration_clipped = clipped_logdet_from_eigenvalues(
        calibration.eigenvalues, margin=1e-7,
    )
    calibration_tsd = float(calibration_tsd_value)
    test_generator = torch.Generator().manual_seed(seed + 10_000_000)
    test_x = torch.randn(samples, 1, generator=test_generator, dtype=torch.float64)
    test_y = test_x[:, None, :] + math.sqrt(noise) * torch.randn(samples, views, 1, generator=test_generator, dtype=torch.float64)
    test_f = hermite(test_x, dimension); test_g = hermite(test_y.reshape(-1, 1) / math.sqrt(1.0 + noise), dimension).reshape(samples, views, dimension)
    heldout = evaluate_heldout_spectrum(test_f, test_g, calibration)
    raw_eigenvalues = heldout.eigenvalues
    tsd_value, clipped = clipped_logdet_from_eigenvalues(raw_eigenvalues, margin=1e-7)
    tsd = float(tsd_value)
    truth = exact_tsd(noise, dimension)
    return {
        "heldout_tsd": tsd, "calibration_tsd": calibration_tsd,
        "train_test_gap": calibration_tsd - tsd,
        "exact_tsd": truth, "absolute_error": abs(tsd - truth),
        "trace_dependence": float(raw_eigenvalues.sum()),
        "heldout_singular_values": heldout.singular_values.tolist(),
        "heldout_eigenvalues": raw_eigenvalues.tolist(),
        "diagonal_correlation_diagnostic": heldout.diagonal_correlations.tolist(),
        "raw_largest_eigenvalue": float(raw_eigenvalues.max()),
        "clipped_mode_count": clipped,
        "calibration_clipped_mode_count": calibration_clipped,
    }


def regression(records: list[dict[str, object]]) -> dict[str, float]:
    x = torch.tensor([float(item["exact_tsd"]) for item in records], dtype=torch.float64)
    y = torch.tensor([float(item["heldout_tsd"]) for item in records], dtype=torch.float64)
    design = torch.stack((x, torch.ones_like(x)), 1)
    coefficients = torch.linalg.lstsq(design, y).solution
    prediction = design @ coefficients
    residual = ((y - prediction) ** 2).sum(); total = ((y - y.mean()) ** 2).sum()
    x_rank = torch.argsort(torch.argsort(x)).double(); y_rank = torch.argsort(torch.argsort(y)).double()
    spearman = float(torch.corrcoef(torch.stack((x_rank, y_rank)))[0, 1])
    grouped: dict[tuple[int, float], list[float]] = {}
    for item in records:
        grouped.setdefault((int(item["replicate"]) % 2, float(item["noise_variance"])), []).append(float(item["heldout_tsd"]))
    noises = sorted({float(item["noise_variance"]) for item in records})
    halves = []
    for parity in (0, 1): halves.append(torch.tensor([sum(grouped[(parity, noise)]) / len(grouped[(parity, noise)]) for noise in noises], dtype=torch.float64))
    reliability = float(torch.corrcoef(torch.stack(halves))[0, 1])
    violations = comparisons = 0
    for replicate in sorted({int(item["replicate"]) for item in records}):
        sequence = sorted((float(item["noise_variance"]), float(item["heldout_tsd"])) for item in records if int(item["replicate"]) == replicate)
        for left, right in zip(sequence, sequence[1:]):
            comparisons += 1; violations += int(right[1] > left[1])
    gaps = [abs(float(item["train_test_gap"])) for item in records]
    return {
        "slope": float(coefficients[0]), "intercept": float(coefficients[1]),
        "r_squared": float(1.0 - residual / total), "spearman": spearman,
        "mae": float((x - y).abs().mean()), "test_retest_reliability": reliability,
        "monotonicity_violations": violations, "monotonicity_comparisons": comparisons,
        "monotonicity_violation_rate": violations / max(1, comparisons),
        "absolute_train_test_gap_mean": sum(gaps) / len(gaps),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--sample-sizes", default="512,2048,8192")
    parser.add_argument("--seed", type=int, default=20271000)
    args = parser.parse_args()
    sample_sizes = tuple(int(value) for value in args.sample_sizes.split(",") if value)
    if not sample_sizes or any(value < 2 for value in sample_sizes): raise ValueError("invalid --sample-sizes")
    records = []
    noises = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
    for samples in sample_sizes:
        for views in (1, 8):
            for noise_index, noise in enumerate(noises):
                for replicate in range(args.replicates):
                    values = estimate(noise, samples, views, args.feature_dim, args.seed + samples * 100 + views * 10000 + noise_index * 1000 + replicate)
                    records.append({"samples": samples, "views": views, "noise_variance": noise, "replicate": replicate, **values})
    summaries = {}
    for samples in sample_sizes:
        for views in (1, 8):
            selected = [item for item in records if item["samples"] == samples and item["views"] == views]
            summaries[f"n{samples}_m{views}"] = regression(selected)
    chain = []
    cumulative_noise = 0.0
    for stage, increment in enumerate((0.1, 0.2, 0.5, 1.0), 1):
        cumulative_noise += increment
        values = estimate(cumulative_noise, 8192, 8, args.feature_dim, args.seed + 9_000_000 + stage)
        chain.append({"stage": stage, "incremental_noise": increment, "cumulative_noise": cumulative_noise, **values})
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "heldout_spectral_method": "svdvals_of_full_canonical_cross_operator",
        "parameters": vars(args),
        "calibration_summary": summaries,
        "data_processing_chain": chain,
        "records": records,
    }
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "e7_tsd_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    if os.environ.get("FMCA_HARNESS_RUN_DIR"):
        with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "e7_tsd_calibration", "conditions": len(records)}) + "\n")
    print(json.dumps({"calibration_summary": summaries, "data_processing_chain": chain}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
