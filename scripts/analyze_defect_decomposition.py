"""E-A1(a): report the defect ratio's numerator and denominator separately.

The audit's first item: a normalized defect can fall because the
numerator shrank (genuine closure) or because the denominator grew or
shrank for unrelated reasons (a weak or a large endpoint operator).
Every unit.json already carries both terms plus the endpoint spectrum,
so this needs no compute -- only the discipline of printing them.

The degenerate-encoder control rows, which need forward passes, are a
separate GPU job; this script covers the decomposition half.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SETS = [("CIFAR-10 gate v8", "results/gate1/gate1_20260820_v8/units", range(1, 6)),
        ("CIFAR-100 pilot", "results/gate1/gate1_20260821_c100pilot/units", range(1, 6)),
        ("CIFAR-10 star control", "results/gate1/gate1_20260821_v8_parallel/units", range(1, 4)),
        ("CIFAR-100 star control", "results/gate1/gate1_20260822_c100_parallel/units", range(1, 4))]
ROWS = ["product_endpoint", "additive_mview", "additive_2view",
        "final_mview", "final_2view", "amdim_cross", "product_only"]


def entropy_effective_rank(values):
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    probabilities = values / values.sum()
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def unit_row(path):
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record.get("status") != "complete":
        return None
    certificate = record["certificate"]
    point = certificate["point"]
    endpoint = point["endpoint_singular_values"]
    return {
        "ratio": certificate["normalized_closure_defect"],
        "numerator": point["delta_frobenius"],
        "denominator": certificate["dir_frobenius"],
        "dir_top": max(endpoint),
        "dir_rank": entropy_effective_rank(endpoint),
        "probe": record["linear_probe"]["test_accuracy"] * 100,
    }


def main() -> None:
    payload = {}
    for title, root, seeds in SETS:
        rows = []
        for variant in ROWS:
            values = [v for seed in seeds
                      if (v := unit_row(ROOT / root / f"{variant}__seed{seed}" / "unit.json"))]
            if values:
                rows.append((variant, values))
        if not rows:
            continue
        print(f"\n=== {title} ===")
        print(f"{'row':20s} {'n':>2s} {'ratio':>8s} {'numerator':>10s} {'denominator':>12s} "
              f"{'dir top':>8s} {'dir effrank':>12s} {'probe':>7s}")
        for variant, values in rows:
            def column(key):
                return np.array([v[key] for v in values])
            print(f"{variant:20s} {len(values):>2d} "
                  f"{column('ratio').mean():>8.3f} {column('numerator').mean():>10.3f} "
                  f"{column('denominator').mean():>12.3f} {column('dir_top').mean():>8.3f} "
                  f"{column('dir_rank').mean():>12.2f} {column('probe').mean():>7.2f}")
            payload.setdefault(title, {})[variant] = {
                key: float(np.mean([v[key] for v in values]))
                for key in ("ratio", "numerator", "denominator", "dir_top", "dir_rank", "probe")
            }
        # The audit's actual question: does the ratio track the numerator,
        # or is it riding the denominator?
        ratios = np.array([np.mean([v["ratio"] for v in values]) for _, values in rows])
        numerators = np.array([np.mean([v["numerator"] for v in values]) for _, values in rows])
        denominators = np.array([np.mean([v["denominator"] for v in values]) for _, values in rows])
        if len(ratios) > 2:
            print(f"  across rows: corr(ratio, numerator) = {np.corrcoef(ratios, numerators)[0, 1]:+.3f}, "
                  f"corr(ratio, denominator) = {np.corrcoef(ratios, denominators)[0, 1]:+.3f}")

    out = ROOT / "results/gate1/DEFECT_DECOMPOSITION_20260824.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
