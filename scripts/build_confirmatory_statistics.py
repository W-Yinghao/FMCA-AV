#!/usr/bin/env python3
"""Build paired effect-size, bootstrap-CI and Holm-corrected confirmatory assets."""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
import random
import re
import statistics


SEED_PATTERN = re.compile(r"(?:^|[-_])seed[-_]?([0-9]+)(?:[-_]|$)")
PILOT_TOKENS = ("smoke", "screening", "pilot")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def seed(row: dict[str, str]) -> str:
    if row.get("seed", ""):
        return str(row["seed"])
    match = SEED_PATTERN.search(str(row.get("name", "")).lower())
    return match.group(1) if match else ""


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values); position = probability * (len(ordered) - 1)
    lower = int(math.floor(position)); upper = int(math.ceil(position)); fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_interval(values: list[float], seed_value: int, replicates: int = 10000) -> tuple[float, float]:
    generator = random.Random(seed_value); means = []
    for _ in range(replicates):
        means.append(statistics.fmean(values[generator.randrange(len(values))] for _ in values))
    return percentile(means, 0.025), percentile(means, 0.975)


def sign_flip_p(values: list[float]) -> float:
    nonzero = [value for value in values if value != 0.0]
    if not nonzero:
        return 1.0
    observed = abs(statistics.fmean(nonzero)); count = 0; total = 2 ** len(nonzero)
    if len(nonzero) <= 20:
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
            count += abs(statistics.fmean(sign * value for sign, value in zip(signs, nonzero))) >= observed - 1e-15
        return count / total
    generator = random.Random(20309000); total = 200000
    for _ in range(total):
        count += abs(statistics.fmean(value if generator.random() < 0.5 else -value for value in nonzero)) >= observed - 1e-15
    return (count + 1) / (total + 1)


def add_test(output: list[dict[str, object]], family: str, experiment: str, dataset: str,
             metric: str, contrast: str, values: list[float], pair_ids: list[str]) -> None:
    if not values:
        return
    mean = statistics.fmean(values); standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    low, high = bootstrap_interval(values, 20309000 + sum(ord(value) for value in family + experiment + dataset + metric + contrast))
    dz: object = mean / standard_deviation if standard_deviation > 0 else ""
    hedges_gz: object = dz * (1.0 - 3.0 / (4.0 * len(values) - 5.0)) if dz != "" and len(values) > 1 else ""
    output.append({
        "claim_family": family, "experiment": experiment, "dataset": dataset, "metric": metric,
        "contrast": contrast, "paired_n": len(values), "pair_ids": ";".join(pair_ids),
        "paired_difference_mean": mean, "paired_difference_std": standard_deviation,
        "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
        "cohens_dz": dz, "hedges_gz": hedges_gz, "exact_sign_flip_p": sign_flip_p(values),
        "holm_adjusted_p": "", "confirmatory_ready": len(values) >= 3,
    })


def paired_reference_tests(rows: list[dict[str, str]], output: list[dict[str, object]], family: str,
                           experiment: str, metric: str, group_fields: tuple[str, ...], reference: str = "fmca_av") -> None:
    grouped: dict[tuple[str, ...], dict[str, dict[str, float]]] = {}
    for row in rows:
        method = str(row.get("method", "")); pair = seed(row)
        if (not method or not pair or row.get(metric, "") == ""
                or any(token in str(row.get("name", "")).lower() for token in PILOT_TOKENS)):
            continue
        key = tuple(str(row.get(field, "")) for field in group_fields)
        grouped.setdefault(key, {}).setdefault(method, {})[pair] = float(row[metric])
    for key, methods in sorted(grouped.items()):
        if reference not in methods:
            continue
        for comparator, comparator_values in sorted(methods.items()):
            if comparator == reference:
                continue
            pairs = sorted(set(methods[reference]) & set(comparator_values))
            values = [methods[reference][pair] - comparator_values[pair] for pair in pairs]
            dataset = key[0] if key else ""
            context = ",".join(f"{field}={value}" for field, value in zip(group_fields[1:], key[1:]))
            add_test(output, family, experiment, dataset, metric,
                     f"{reference} minus {comparator}" + (f" ({context})" if context else ""), values, pairs)


def matched_compute_tests(output: list[dict[str, object]]) -> None:
    rows = read_csv(Path("results/e5/matched_compute_runs.csv")); grouped: dict[tuple[str, ...], list[tuple[str, float]]] = {}
    for row in rows:
        if row.get("paired_accuracy_difference_v8_minus_v2", "") == "":
            continue
        key = tuple(str(row.get(field, "")) for field in ("dataset", "method", "backbone", "aggregation"))
        grouped.setdefault(key, []).append((seed(row) or str(row.get("key", "")), float(row["paired_accuracy_difference_v8_minus_v2"])))
    for key, values in sorted(grouped.items()):
        add_test(output, "C2/C3", "E5-matched-compute", key[0], "accuracy",
                 f"V8 minus V2 (method={key[1]},backbone={key[2]},aggregation={key[3]})",
                 [value for _, value in values], [pair for pair, _ in values])


def e4_aggregation_tests(output: list[dict[str, object]]) -> None:
    rows = read_csv(Path("results/e4/aggregation_ablation.csv"))
    grouped: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for row in rows:
        pair = seed(row); aggregation = str(row.get("aggregation", ""))
        score = row.get("best_validation_score", "")
        if not pair or not aggregation or score == "":
            continue
        key = (str(row.get("dataset", "")), str(row.get("views", "")))
        grouped.setdefault(key, {}).setdefault(aggregation, {})[pair] = float(score)
    for (dataset, views), aggregations in sorted(grouped.items()):
        reference = aggregations.get("mean", {})
        for comparator in ("first", "deepsets", "concat"):
            comparator_values = aggregations.get(comparator, {})
            pairs = sorted(set(reference) & set(comparator_values))
            add_test(output, "C2", "E4-aggregation", dataset, "validation_dependence_score",
                     f"mean minus {comparator} (views={views})",
                     [reference[pair] - comparator_values[pair] for pair in pairs], pairs)


def factor_tests(output: list[dict[str, object]]) -> None:
    rows = read_csv(Path("results/e7/factor_probe_summary.csv")); grouped: dict[tuple[str, ...], list[tuple[str, float]]] = {}
    for row in rows:
        status_path = Path("runs") / str(row.get("run_id", "")) / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if any(token in str(status.get("name", "")).lower() for token in PILOT_TOKENS):
                continue
        for metric, contrast in (("mean_top_minus_random", "eigen_top minus random"),
                                 ("mean_top_minus_eigen_bottom", "eigen_top minus eigen_bottom"),
                                 ("mean_top_minus_pca_top", "eigen_top minus pca_top")):
            if row.get(metric, "") == "":
                continue
            key = (row.get("dataset", ""), row.get("channel", ""), row.get("factor_name", ""), metric, contrast)
            grouped.setdefault(tuple(map(str, key)), []).append((str(row.get("run_id", "")), float(row[metric])))
    for key, values in sorted(grouped.items()):
        add_test(output, "C4", "E7-factor-ranking", key[0], key[3],
                 f"{key[4]} (channel={key[1]},factor={key[2]})", [value for _, value in values], [pair for pair, _ in values])


def holm(rows: list[dict[str, object]]) -> None:
    families: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if bool(row["confirmatory_ready"]):
            families.setdefault(str(row["claim_family"]), []).append(row)
    for values in families.values():
        ordered = sorted(values, key=lambda row: float(row["exact_sign_flip_p"])); previous = 0.0; count = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = max(previous, min(1.0, (count - rank) * float(row["exact_sign_flip_p"])))
            row["holm_adjusted_p"] = adjusted; previous = adjusted


def main() -> int:
    rows: list[dict[str, object]] = []
    paired_reference_tests(read_csv(Path("results/e5/matched_ssl_runs.csv")), rows, "C3", "E5-matched-SSL", "accuracy", ("dataset", "views", "architecture", "protocol"))
    paired_reference_tests(read_csv(Path("results/e6/robustness_table.csv")), rows, "C3", "E6-robustness", "mean_accuracy", ("dataset", "suite"))
    paired_reference_tests(read_csv(Path("results/e6/generalization_transfer_table.csv")), rows, "C3", "E6-generalization-transfer", "primary_value", ("dataset", "protocol", "label_fraction", "task", "primary_metric"))
    for metric in ("top20_mask_iou", "pixel_auprc", "pointing_game", "foreground_energy_ratio", "top_faithfulness_auc_gap"):
        paired_reference_tests(read_csv(Path("results/e9/localization_table.csv")), rows, "C7", "E9-localization", metric, ("dataset", "architecture", "map"))
    e4_aggregation_tests(rows); matched_compute_tests(rows); factor_tests(rows); holm(rows)
    output = Path("results/statistics"); output.mkdir(parents=True, exist_ok=True)
    fields = ["claim_family", "experiment", "dataset", "metric", "contrast", "paired_n", "pair_ids",
              "paired_difference_mean", "paired_difference_std", "bootstrap_ci95_low", "bootstrap_ci95_high",
              "cohens_dz", "hedges_gz", "exact_sign_flip_p", "holm_adjusted_p", "confirmatory_ready"]
    temporary = output / "confirmatory_tests.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(output / "confirmatory_tests.csv")
    temporary = output / "confirmatory_tests.json.tmp"
    temporary.write_text(json.dumps({"tests": rows, "count": len(rows)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output / "confirmatory_tests.json")
    caption = ("Paired confirmatory comparisons use frozen seed pairs. Differences are reference minus comparator, "
               "with deterministic seed bootstrap intervals, paired Cohen/Hedges effect sizes, exact two-sided sign-flip tests, "
               "and Holm correction within each claim family. Rows with fewer than three pairs are retained but marked not confirmatory-ready.\n")
    temporary = output / "confirmatory_statistics_caption.txt.tmp"; temporary.write_text(caption, encoding="utf-8")
    temporary.replace(output / "confirmatory_statistics_caption.txt")
    print(json.dumps({"tests": len(rows), "claim_families": sorted({row["claim_family"] for row in rows})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
