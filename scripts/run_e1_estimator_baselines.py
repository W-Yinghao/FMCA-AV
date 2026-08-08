#!/usr/bin/env python3
"""Validation-tuned Nyström/KICA/HSIC controls for Gaussian operator recovery."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION, fit_spectral_calibration


def hermite(values: torch.Tensor, dimension: int) -> torch.Tensor:
    features = []
    previous = torch.ones_like(values)
    current = values
    for degree in range(1, dimension + 1):
        features.append(current)
        following = (values * current - math.sqrt(degree) * previous) / math.sqrt(degree + 1)
        previous, current = current, following
    return torch.cat(features, dim=1)


def nystrom(values: torch.Tensor, landmarks: torch.Tensor, bandwidth: float) -> torch.Tensor:
    squared = (values[:, None, :] - landmarks[None, :, :]).square().sum(2)
    return torch.exp(-squared / (2.0 * bandwidth * bandwidth))


def rff(values: torch.Tensor, frequencies: torch.Tensor, phases: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return math.sqrt(2.0 / frequencies.shape[1]) * torch.cos(values @ (frequencies / bandwidth) + phases)


def spectrum(left: torch.Tensor, right: torch.Tensor, ridge: float) -> torch.Tensor:
    return fit_spectral_calibration(left, right[:, None, :], ridge=ridge, centered=True).eigenvalues


def normalized_hsic(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left - left.mean(0, keepdim=True)
    right = right - right.mean(0, keepdim=True)
    denominator = max(1, len(left) - 1)
    cross = left.T @ right / denominator
    left_covariance = left.T @ left / denominator
    right_covariance = right.T @ right / denominator
    scale = (left_covariance.square().sum() * right_covariance.square().sum()).sqrt().clamp_min(1e-12)
    return float(cross.square().sum() / scale)


def sample(count: int, rho: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    left = torch.randn(count, 1, generator=generator, dtype=torch.float64)
    noise = torch.randn(count, 1, generator=generator, dtype=torch.float64)
    right = rho * left + math.sqrt(1.0 - rho * rho) * noise
    return left, right


def errors(estimate: torch.Tensor, truth: torch.Tensor, prefix: str = "") -> dict[str, float]:
    count = min(len(estimate), len(truth))
    delta = (estimate[:count] - truth[:count]).abs()
    return {
        prefix + "spectrum_mae": float(delta.mean()),
        prefix + "spectrum_relative_l1": float(delta.sum() / truth[:count].sum().clamp_min(1e-12)),
        prefix + "trace_error": float((estimate[:count].sum() - truth[:count].sum()).abs()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20301000)
    args = parser.parse_args()
    sample_sizes = (500, 2000, 10000)
    correlations = (0.3, 0.6, 0.9)
    bandwidths = (0.25, 0.5, 1.0, 2.0)
    ridges = (1e-2, 1e-3, 1e-4)
    dimension = 8
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for rho_index, rho in enumerate(correlations):
        truth = torch.tensor([rho ** (2 * degree) for degree in range(1, dimension + 1)], dtype=torch.float64)
        for samples in sample_sizes:
            for replicate in range(args.replicates):
                generator = torch.Generator().manual_seed(args.seed + rho_index * 100000 + samples * 10 + replicate)
                train_x, train_y = sample(samples, rho, generator)
                validation_x, validation_y = sample(5000, rho, generator)
                test_x, test_y = sample(10000, rho, generator)
                common = {"rho": rho, "samples": samples, "replicate": replicate}

                raw_estimate = spectrum(train_x, train_y, 1e-3)
                raw_test = spectrum(test_x, test_y, 1e-3)
                records.append({**common, "method": "linear_cca", "selected_bandwidth": None,
                                "selected_ridge": 1e-3, "hsic": None,
                                **errors(raw_estimate, truth), **errors(raw_test, truth, "test_")})
                train_hermite_x, train_hermite_y = hermite(train_x, dimension), hermite(train_y, dimension)
                test_hermite_x, test_hermite_y = hermite(test_x, dimension), hermite(test_y, dimension)
                hermite_estimate = spectrum(train_hermite_x, train_hermite_y, 1e-3)
                hermite_test = spectrum(test_hermite_x, test_hermite_y, 1e-3)
                records.append({**common, "method": "hermite_operator", "selected_bandwidth": None,
                                "selected_ridge": 1e-3, "hsic": None,
                                **errors(hermite_estimate, truth), **errors(hermite_test, truth, "test_")})

                landmark_count = min(64, samples)
                landmarks = train_x[:landmark_count].clone()
                frequencies = torch.randn(1, 64, generator=generator, dtype=torch.float64)
                phases = 2.0 * math.pi * torch.rand(64, generator=generator, dtype=torch.float64)
                for method in ("nystrom", "rff_kica"):
                    best: tuple[float, float, float] | None = None
                    for bandwidth in bandwidths:
                        if method == "nystrom":
                            val_left = nystrom(validation_x, landmarks, bandwidth)
                            val_right = nystrom(validation_y, landmarks, bandwidth)
                        else:
                            val_left = rff(validation_x, frequencies, phases, bandwidth)
                            val_right = rff(validation_y, frequencies, phases, bandwidth)
                        for ridge in ridges:
                            validation_score = float(spectrum(val_left, val_right, ridge)[:dimension].sum())
                            candidate = (validation_score, bandwidth, ridge)
                            if best is None or candidate[0] > best[0]:
                                best = candidate
                    assert best is not None
                    _, bandwidth, ridge = best
                    if method == "nystrom":
                        train_left = nystrom(train_x, landmarks, bandwidth)
                        train_right = nystrom(train_y, landmarks, bandwidth)
                        test_left = nystrom(test_x, landmarks, bandwidth)
                        test_right = nystrom(test_y, landmarks, bandwidth)
                    else:
                        train_left = rff(train_x, frequencies, phases, bandwidth)
                        train_right = rff(train_y, frequencies, phases, bandwidth)
                        test_left = rff(test_x, frequencies, phases, bandwidth)
                        test_right = rff(test_y, frequencies, phases, bandwidth)
                    estimate = spectrum(train_left, train_right, ridge)
                    test_estimate = spectrum(test_left, test_right, ridge)
                    records.append({**common, "method": method, "selected_bandwidth": bandwidth,
                                    "selected_ridge": ridge, "validation_trace": best[0], "hsic": None,
                                    **errors(estimate, truth), **errors(test_estimate, truth, "test_")})

                hsic_left = rff(train_x, frequencies, phases, 1.0)
                hsic_right = rff(train_y, frequencies, phases, 1.0)
                test_hsic_left = rff(test_x, frequencies, phases, 1.0)
                test_hsic_right = rff(test_y, frequencies, phases, 1.0)
                records.append({**common, "method": "normalized_hsic", "selected_bandwidth": 1.0,
                                "selected_ridge": None, "hsic": normalized_hsic(hsic_left, hsic_right),
                                "test_hsic": normalized_hsic(test_hsic_left, test_hsic_right),
                                "spectrum_mae": None, "spectrum_relative_l1": None, "trace_error": None,
                                "test_spectrum_mae": None, "test_spectrum_relative_l1": None,
                                "test_trace_error": None})
    payload = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "parameters": vars(args), "selection": "maximize top-8 validation trace on an independent validation split",
        "estimation_protocol": "primary spectrum is estimated from the requested-size training split",
        "test_protocol": "independent 10,000-sample test split is reported only as an out-of-sample diagnostic",
        "records": records,
        "runtime_seconds": time.perf_counter() - started,
    }
    output = (
        Path(args.output) if args.output else
        Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / "e1_estimator_baselines.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    run_dir = os.environ.get("FMCA_HARNESS_RUN_DIR")
    if run_dir:
        with (Path(run_dir) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "e1_estimator_baselines", "conditions": len(records)}) + "\n")
    print(json.dumps({"conditions": len(records), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
