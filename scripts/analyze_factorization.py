"""Does endpoint dependence factorize through the path? (existing data)

Tests the factorization claim on records we already have, with no new
compute.  Two statistics, both stored in every unit.json:

  alignment        cos between C_dir and C_comp as matrices; scale-free
  principal angles between the top singular subspaces of C_dir and
                   C_comp -- the non-circular question, "do the
                   endpoint's dominant directions lie inside the
                   path-supported subspace?"

A third, the certified fraction sum(s_cert)/sum(sigma(C_dir)), is
reported but flagged: s_cert = [sigma(C_comp) - delta_op]_+ subtracts
the defect, so a unit with a smaller defect scores higher partly by
construction.  It is NOT evidence on its own; the angles are.

Also prints the naive sigma-product against the true composed top
singular value, which is the empirical form of the red line that
forbids sigma-products.
"""

import json
from pathlib import Path
import math

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 8

SETS = [
    ("CIFAR-10 (gate v8)", "results/gate1/gate1_20260820_v8/units", (1, 2, 3)),
    ("CIFAR-100 (pilot)", "results/gate1/gate1_20260821_c100pilot/units", (1, 2, 3)),
]
ROWS = [("product_endpoint", "V7 explicit closure"),
        ("additive_mview", "additive (path-blind)"),
        ("final_mview", "flat M-view")]


def unit_statistics(path):
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record.get("status") != "complete":
        return None
    point = record["certificate"]["point"]
    endpoint = np.array(point["endpoint_singular_values"])
    certified = np.array(point["certified_spectrum"])
    left = np.degrees(np.array(point["principal_angles_left"])[:TOP_K])
    right = np.degrees(np.array(point["principal_angles_right"])[:TOP_K])
    return {
        "alignment": point["alignment"],
        "angle_left": float(left.mean()),
        "angle_right": float(right.mean()),
        "certified_fraction": float(certified.sum() / max(endpoint.sum(), 1e-12)),
        "naive_sigma_product": max(naive) if isinstance(
            (naive := point.get("naive_sigma_product")), list) else naive,
        "path_top": float(max(point["path_singular_values"])),
    }


def main() -> None:
    for title, root, seeds in SETS:
        print(f"\n=== {title} ===")
        print(f"{'row':24s} {'n':>2s} {'alignment':>10s} {'angle L':>9s} {'angle R':>9s} "
              f"{'cert.frac':>10s}")
        for variant, label in ROWS:
            values = [s for seed in seeds
                      if (s := unit_statistics(ROOT / root / f"{variant}__seed{seed}" / "unit.json"))]
            if not values:
                continue
            def column(key):
                return np.array([v[key] for v in values])
            print(f"{label:24s} {len(values):2d} "
                  f"{column('alignment').mean():10.3f} "
                  f"{column('angle_left').mean():8.1f}° "
                  f"{column('angle_right').mean():8.1f}° "
                  f"{column('certified_fraction').mean():10.3f}")
            print(f"{'  per seed alignment':24s}    " +
                  "  ".join(f"{x:.3f}" for x in column("alignment")))
            print(f"{'  per seed angle L':24s}    " +
                  "  ".join(f"{x:.1f}°" for x in column("angle_left")))

    print("\n=== red line: naive sigma-product vs true composed top (CIFAR-10) ===")
    for variant, label in ROWS:
        stats = unit_statistics(
            ROOT / SETS[0][1] / f"{variant}__seed1" / "unit.json")
        if stats and stats["naive_sigma_product"] is not None:
            ratio = stats["naive_sigma_product"] / max(stats["path_top"], 1e-12)
            print(f"{label:24s} naive={stats['naive_sigma_product']:.3f}  "
                  f"true={stats['path_top']:.3f}  ratio={ratio:.2f}x")

    print("\nAngles are the non-circular statistic: smaller = the endpoint's")
    print("dominant directions lie inside the path-supported subspace.")


if __name__ == "__main__":
    main()
