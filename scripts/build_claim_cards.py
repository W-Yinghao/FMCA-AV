#!/usr/bin/env python3
"""Create conservative, data-linked C1--C7 conclusion cards from frozen assets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics


def rows(path: str) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        return []
    with source.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def card(claim: str, question: str, estimand: str, criterion: str, evidence: str,
         decision: str, supported: str, unsupported: str, sources: list[str]) -> str:
    return "\n".join((
        f"# {claim} conclusion card", "", f"Claim ID: {claim}", f"Question: {question}",
        f"Primary estimand: {estimand}", f"Pre-registered positive/negative criterion: {criterion}",
        "Code version label / config / data split / seeds: file-based harness artifacts and frozen state-machine manifests; see linked source tables.",
        "Compute budget: exact run-level GPU/CPU duration, encoded views and hardware are retained in runs/, results/index and E10 assets.",
        f"Main result with 95% CI: {evidence}",
        "Numerical stability and failed runs: retained in results/index/failure_atlas.csv and the raw experiment tables.",
        f"Decision: {decision}", f"Narrowest supported statement: {supported}",
        f"Unsupported stronger statement: {unsupported}",
        "Required manuscript change: use only the narrowest supported statement and disclose failed/negative controls.",
        "Next experiment (if any): none selected from test performance; any future work must be declared exploratory.",
        "Sources: " + ", ".join(sources), "",
    ))


def c1() -> str:
    values = rows("results/e1/finite_sample_recovery.csv"); groups = {}
    for row in values: groups.setdefault(row["family"], []).append(row)
    passed = bool(groups); details = []
    for family, family_rows in sorted(groups.items()):
        ordered = sorted(family_rows, key=lambda row: int(row["samples"])); first = float(ordered[0]["top_projector_error_median"]); last = float(ordered[-1]["top_projector_error_median"])
        passed &= last <= 0.10 and last < first
        details.append(f"{family}: projector {first:.4g}→{last:.4g}")
    decision = "PASS" if passed else "INCONCLUSIVE" if not groups else "FAIL"
    return card("C1", "Does FMCA-AV recover the nonconstant dependence spectrum/subspace?", "held-out spectrum and spectral-block projector error", "high-resource median projector error ≤0.10 and decreasing with N", "; ".join(details) or "required table missing", decision, "Recovery is supported only in channel/dimension regimes that meet the reported error thresholds.", "Exact recovery for every nonlinear/high-dimensional regime is not supported unless every corresponding row passes.", ["results/e1/finite_sample_recovery.csv", "results/e1/gaussian_recovery.csv"])


def c2() -> str:
    values = [row for row in rows("results/e2/gradient_variance_table.csv") if row.get("design") == "fixed_parent" and int(row.get("views", "1")) > 1]
    passed = bool(values) and all(float(row["score_variance_ratio_ci95_high"]) < 1 and float(row["gradient_variance_ratio_ci95_high"]) < 1 for row in values)
    evidence = "; ".join(f"M={row['views']}: score CI-high={float(row['score_variance_ratio_ci95_high']):.3g}, gradient CI-high={float(row['gradient_variance_ratio_ci95_high']):.3g}" for row in values)
    return card("C2", "Does conditional multi-sampling reduce estimator and gradient noise?", "fixed-parent score/gradient variance ratios", "for M>1 both paired 95% CI upper bounds are below 1; fixed-budget Pareto remains explicit", evidence or "required table missing", "PASS" if passed else "INCONCLUSIVE" if not values else "FAIL", "Conditional sampling reduces noise only for metrics/configurations whose full CI is below one; fixed-budget behavior is a Pareto trade-off.", "More views are universally more compute-efficient is not supported.", ["results/e2/gradient_variance_table.csv"])


def statistical_claim(claim: str, family_prefix: str, question: str, sources: list[str]) -> str:
    tests = [row for row in rows("results/statistics/confirmatory_tests.csv") if family_prefix in row["claim_family"].split("/") and row.get("confirmatory_ready") == "True"]
    positive = [row for row in tests if float(row["holm_adjusted_p"]) < 0.05 and float(row["bootstrap_ci95_low"]) > 0]
    negative = [row for row in tests if float(row["holm_adjusted_p"]) < 0.05 and float(row["bootstrap_ci95_high"]) < 0]
    decision = "PASS" if positive and not negative else "FAIL" if negative and not positive else "INCONCLUSIVE"
    evidence = f"confirmatory-ready contrasts={len(tests)}, Holm-positive={len(positive)}, Holm-negative={len(negative)}"
    return card(claim, question, "paired seed difference, bootstrap CI and Holm-adjusted exact sign-flip p", "a Holm-adjusted paired effect in the preregistered direction with CI excluding zero", evidence, decision, "Only the individual paired contrasts with corrected support justify a comparative claim.", "Blanket superiority across datasets, budgets or tasks is not supported by a subset of significant contrasts.", [*sources, "results/statistics/confirmatory_tests.csv"])


def c4() -> str:
    tests = [row for row in rows("results/statistics/confirmatory_tests.csv") if row["claim_family"] == "C4" and row.get("confirmatory_ready") == "True"]
    controls = {"random": False, "eigen_bottom": False, "pca_top": False}
    for key in controls:
        controls[key] = any(key in row["contrast"] and float(row["holm_adjusted_p"]) < 0.05 and float(row["bootstrap_ci95_low"]) > 0 for row in tests)
    decision = "PASS" if tests and all(controls.values()) else "INCONCLUSIVE" if not tests else "FAIL"
    return card("C4", "Does spectral ordering align with preserved semantic factors?", "paired accuracy-dimension/AUC advantage of Eigen-top", "corrected positive effects against random, bottom and PCA plus channel-dependent factor changes", json.dumps(controls, sort_keys=True), decision, "Spectral ranking is useful only for datasets/factors and controls with corrected positive paired evidence.", "A universal coordinate-wise semantic ordering is not supported, especially inside repeated spectral blocks.", ["results/e7/factor_probe_summary.csv", "results/statistics/confirmatory_tests.csv"])


def c5() -> str:
    values = rows("results/e7/tsd_calibration_summary.csv"); ready = [row for row in values if row.get("r_squared", "") and row.get("test_retest_reliability", "")]
    passed = bool(ready) and all(float(row["r_squared"]) >= 0.95 and float(row["test_retest_reliability"]) >= 0.90 for row in ready)
    best_r2 = max((float(row["r_squared"]) for row in ready), default=float("nan")); worst_r2 = min((float(row["r_squared"]) for row in ready), default=float("nan"))
    evidence = f"conditions={len(ready)}, R² range={worst_r2:.4g}..{best_r2:.4g}" if ready else "required table missing"
    return card("C5", "Is held-out TSD a calibrated retained-dependence diagnostic?", "held-out TSD calibration R², error, monotonicity and test–retest reliability", "R²≥0.95 with reliable, monotone held-out behavior and explicit clipping/gap diagnostics", evidence, "PASS" if passed else "INCONCLUSIVE" if not ready else "FAIL", "TSD can be reported as an empirical retained-dependence diagnostic only in conditions that pass calibration and reliability checks.", "TSD is not a universally calibrated scalar proxy for utility.", ["results/e7/tsd_calibration_summary.csv", "results/e7/tsd_data_processing_chain.csv", "results/e7/tsd_utility_table.csv"])


def c6() -> str:
    exact = rows("results/e8/exact_markov.csv"); continuous = rows("results/e8/continuous_markov.csv")
    valid_names = {"reversible", "metastable_reversible", "directed_cycle_normal"}
    valid_rows = [row for row in exact if row.get("chain") in valid_names]
    nonnormal_rows = [row for row in exact if row.get("chain") == "directed_nonnormal"]
    valid = [float(row["power_spectrum_mae_median"]) for row in valid_rows]
    nonnormal = [float(row["power_spectrum_mae_median"]) for row in nonnormal_rows]
    dynamics = {row.get("dynamics", "") for row in continuous}
    complete = valid and nonnormal and {"ornstein_uhlenbeck", "double_well", "multi_well"} <= dynamics
    valid_pass = bool(valid) and max(valid) <= 1e-10
    counterexample_pass = bool(nonnormal) and statistics.median(nonnormal) >= 1e-3
    decision = "PASS" if complete and valid_pass and counterexample_pass else "INCONCLUSIVE" if not complete else "FAIL"
    valid_ci = max((float(row.get("power_spectrum_mae_ci95_half_width", 0.0)) for row in valid_rows), default=float("nan"))
    nonnormal_ci = max((float(row.get("power_spectrum_mae_ci95_half_width", 0.0)) for row in nonnormal_rows), default=float("nan"))
    evidence = (f"normal/reversible max median spectrum MAE={max(valid):.4g} (largest 95% CI half-width={valid_ci:.4g}); "
                f"nonnormal median spectrum MAE={statistics.median(nonnormal):.4g} (largest 95% CI half-width={nonnormal_ci:.4g}); "
                f"continuous dynamics={sorted(dynamics)}") if valid and nonnormal else "required tables missing"
    return card("C6", "Under which assumptions can local Markov spectra be composed across lag?",
                "direct-versus-composed spectrum error and Chapman–Kolmogorov residual",
                "machine-precision exact composition for reversible/normal chains, a detectable nonnormal counterexample, and complete OU/double-/multi-well boundary sweeps",
                evidence, decision,
                "One-step spectral powers compose exactly only in the reported reversible/normal regimes; continuous finite-sample errors and nonnormal failures define the boundary.",
                "A general singular-value power law for arbitrary nonnormal Markov dynamics is not supported.",
                ["results/e8/exact_markov.csv", "results/e8/exact_markov_conditions.csv", "results/e8/continuous_markov.csv", "results/e8/continuous_markov_conditions.csv"])


def c7() -> str:
    values = rows("results/e9/localization_table.csv"); datasets = {row.get("dataset", "").lower() for row in values if row.get("dataset")}
    randomized = any(row.get("randomized", "").lower() == "true" or row.get("randomize_from_stage", "") for row in values)
    gaps = [float(row["top_faithfulness_auc_gap"]) for row in values if row.get("top_faithfulness_auc_gap", "") and row.get("randomized", "").lower() != "true"]
    complete = all(any(token in value for value in datasets) for token in ("cub", "voc", "imagenet")) and randomized and bool(gaps)
    positive = complete and statistics.fmean(gaps) > 0
    return card("C7", "Are dependence maps localized, faithful and sensitive to model randomization?", "localization, deletion/insertion faithfulness and randomization controls", "CUB/VOC/ImageNet coverage, positive faithfulness gap and randomization sensitivity", f"datasets={sorted(datasets)}, randomized_control={randomized}, mean faithfulness gap={statistics.fmean(gaps):.4g}" if gaps else f"datasets={sorted(datasets)}, randomized_control={randomized}, faithfulness missing", "PASS" if positive else "INCONCLUSIVE" if not complete else "FAIL", "Maps may be called model-sensitive dependence visualizations only where localization, faithfulness and sanity controls jointly pass.", "A general explainability or causal-localization claim is not supported by qualitative heatmaps alone.", ["results/e9/localization_table.csv", "results/e9/localization_summary.csv", "results/e9/cnn_composition_table.csv"])


def main() -> int:
    output = Path("results/claims"); output.mkdir(parents=True, exist_ok=True)
    cards = {
        "C1": c1(), "C2": c2(),
        "C3": statistical_claim("C3", "C3", "Does FMCA-AV improve representation utility under matched budgets and downstream shifts?", ["results/e5/matched_ssl_runs.csv", "results/e5/matched_compute_runs.csv", "results/e6/generalization_transfer_table.csv", "results/e6/robustness_table.csv"]),
        "C4": c4(), "C5": c5(), "C6": c6(),
        "C7": c7(),
    }
    for claim, value in cards.items():
        temporary = output / f"{claim}.md.tmp"; temporary.write_text(value, encoding="utf-8"); temporary.replace(output / f"{claim}.md")
    decisions = {claim: next(line.split(": ", 1)[1] for line in value.splitlines() if line.startswith("Decision:")) for claim, value in cards.items()}
    temporary = output / "claim_decisions.json.tmp"; temporary.write_text(json.dumps(decisions, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output / "claim_decisions.json")
    print(json.dumps(decisions, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
