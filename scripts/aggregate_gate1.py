"""Gate 1 aggregation (freeze-before-aggregate; run after ALL 21 units).

Builds the main table and evaluates the preregistered E1/E2/E3 contrasts
from prereg/GATE1_CIFAR10_STRUCTURE_PREREG_FROZEN_20260816.md. Refuses
under-coverage. Collapsed units are reported in place, never dropped.
"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VARIANTS = [
    "final_2view",
    "final_mview",
    "additive_2view",
    "additive_mview",
    "amdim_cross",
    "product_only",
    "product_endpoint",
]
SEEDS = (1, 2, 3)
ADDITIVE_ROWS = ("additive_2view", "additive_mview", "amdim_cross")
DOWNSTREAM_ROWS = ("final_2view", "final_mview", "additive_2view", "additive_mview", "amdim_cross")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="results/gate1/gate1_20260817_v2")
    parser.add_argument("--allow-partial", action="store_true",
                        help="interim look: table only, NO verdicts")
    arguments = parser.parse_args()
    root = Path(arguments.output_root)

    records = {}
    manifest = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            path = root / "units" / f"{variant}__seed{seed}" / "unit.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            if payload.get("status") != "complete":
                continue
            records[(variant, seed)] = payload
            manifest[f"{variant}__seed{seed}"] = hashlib.sha256(path.read_bytes()).hexdigest()

    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    missing = sorted(expected - set(records))
    if missing and not arguments.allow_partial:
        raise SystemExit(f"REFUSING to aggregate: {len(missing)} units missing/incomplete: {missing}")

    def defect(record):
        return record["certificate"]["normalized_closure_defect"]

    rows = []
    for variant in VARIANTS:
        cells = [records[(variant, seed)] for seed in SEEDS if (variant, seed) in records]
        if not cells:
            rows.append({"variant": variant, "seeds": 0})
            continue
        probes = [cell["linear_probe"]["test_accuracy"] * 100 for cell in cells]
        knns = [cell["knn_accuracy"] * 100 for cell in cells]
        defects = [defect(cell) for cell in cells]
        certs = [max(cell["certificate"]["point"]["certified_spectrum"]) for cell in cells]
        ranks = [cell["representation"]["effective_rank"] for cell in cells]
        rows.append(
            {
                "variant": variant,
                "seeds": len(cells),
                "probe_mean": statistics.mean(probes),
                "probe_per_seed": probes,
                "knn_mean": statistics.mean(knns),
                "knn_per_seed": knns,
                "defect_mean": statistics.mean(defects),
                "defect_per_seed": defects,
                "certified_top_mean": statistics.mean(certs),
                "effective_rank_mean": statistics.mean(ranks),
                "collapsed_units": sum(int(cell["collapsed"]) for cell in cells),
            }
        )

    output = {"rows": rows, "missing": [f"{v}/seed{s}" for v, s in missing]}

    if not missing:
        by_variant = {row["variant"]: row for row in rows}
        v7 = by_variant["product_endpoint"]
        pairwise_ok = all(
            records[("product_endpoint", seed)] and
            defect(records[("product_endpoint", seed)]) < defect(records[(row, seed)])
            for row in ADDITIVE_ROWS for seed in SEEDS
        )
        best_additive_defect = min(by_variant[row]["defect_mean"] for row in ADDITIVE_ROWS)
        e1 = pairwise_ok and v7["defect_mean"] < 0.5 * best_additive_defect
        best_probe = max(by_variant[row]["probe_mean"] for row in DOWNSTREAM_ROWS)
        best_knn = max(by_variant[row]["knn_mean"] for row in DOWNSTREAM_ROWS)
        e2 = (
            v7["probe_mean"] >= best_probe - 1.0
            and v7["knn_mean"] >= best_knn - 1.0
        )
        collapsed_rows = [
            row["variant"] for row in rows if row.get("collapsed_units", 0) >= 2
        ]
        output["verdicts"] = {
            "E1_closure_defect": e1,
            "E1_pairwise_all_nine": pairwise_ok,
            "E1_halving": v7["defect_mean"] < 0.5 * best_additive_defect,
            "E1_v7_defect_mean": v7["defect_mean"],
            "E1_best_additive_defect_mean": best_additive_defect,
            "E2_downstream_noninferior": e2,
            "E2_v7_probe_mean": v7["probe_mean"],
            "E2_best_other_probe_mean": best_probe,
            "E2_v7_knn_mean": v7["knn_mean"],
            "E2_best_other_knn_mean": best_knn,
            "E3_collapsed_rows": collapsed_rows,
        }
        (root / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2))

    (root / ("aggregate_partial.json" if missing else "aggregate.json")).write_text(
        json.dumps(output, indent=2)
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
