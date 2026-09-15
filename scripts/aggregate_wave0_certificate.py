"""Wave 0 aggregation (freeze-before-aggregate; run only after ALL units).

Refuses to aggregate under-coverage: the expected unit-key set is rebuilt
from the frozen grids and compared as a PROPER set of unique strings.
Produces the neutral results table with bootstrap CIs and the G1-G6 verdict
rows of the prereg, plus a sha256 manifest of every unit file.
"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_wave0_certificate_suite import (  # noqa: E402
    CERTIFY_THRESHOLD,
    CHILDREN_GRID,
    PARENTS_GRID,
    SEED_GRID,
    group_a_units,
    group_b_units,
)

AGGREGATION_SEED = 20260816
BOOTSTRAP_SAMPLES = 2000


def expected_unit_keys() -> set:
    keys = {key for key, _ in group_a_units()} | {key for key, _ in group_b_units()}
    keys |= {"C/misaligned_diag", "C/rotated_misaligned", "C/nonnormal_markov"}
    return keys


def bootstrap_ci(values, generator):
    if len(values) < 2:
        return (min(values), max(values))
    tensor = torch.tensor(values, dtype=torch.float64)
    indices = torch.randint(0, len(values), (BOOTSTRAP_SAMPLES, len(values)), generator=generator)
    medians = tensor[indices].median(dim=1).values
    return (float(torch.quantile(medians, 0.025)), float(torch.quantile(medians, 0.975)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args()
    root = Path(arguments.output_root)
    unit_dir = root / "units"

    expected = expected_unit_keys()
    records = {}
    manifest = {}
    for path in sorted(unit_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        records[payload["unit_key"]] = payload
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    present = set(records)
    missing = expected - present
    failed = sorted(key for key, record in records.items() if record.get("status") != "complete")
    if missing or failed:
        raise SystemExit(
            f"REFUSING to aggregate: {len(missing)} missing units, {len(failed)} failed units.\n"
            f"missing (first 10): {sorted(missing)[:10]}\nfailed (first 10): {failed[:10]}"
        )

    generator = torch.Generator().manual_seed(AGGREGATION_SEED)
    tables = {}
    for key, record in records.items():
        if key.startswith("C/"):
            continue
        group, name, n_tag, m_tag, _ = key.split("/")
        cell = (group, name, int(n_tag[1:]), int(m_tag[1:]))
        tables.setdefault(cell, []).append(record)

    rows = []
    for cell in sorted(tables):
        group, name, parents, children = cell
        cell_records = tables[cell]
        edge_errors = [record["edge_error_max"] for record in cell_records]
        dir_errors = [record["dir_error_max"] for record in cell_records]
        accepts = [record["certificate_accepts"] for record in cell_records]
        delta_ops = [record["report"]["delta_operator"] for record in cell_records]
        rows.append(
            {
                "group": group,
                "case": name,
                "parents": parents,
                "children": children,
                "seeds": len(cell_records),
                "edge_error_median": statistics.median(edge_errors),
                "edge_error_ci": bootstrap_ci(edge_errors, generator),
                "dir_error_median": statistics.median(dir_errors),
                "dir_error_ci": bootstrap_ci(dir_errors, generator),
                "delta_op_median": statistics.median(delta_ops),
                "delta_op_ci": bootstrap_ci(delta_ops, generator),
                "accept_rate": sum(accepts) / len(accepts),
            }
        )

    verdicts = {}
    positive = {"closed_chain", "full_orders", "truncated"}
    g1_ok = True
    for row in rows:
        certifying_case = row["case"] == "closed_chain" or (
            row["group"] == "B" and row["case"] == "full_orders"
        )
        if certifying_case and row["accept_rate"] < 1.0:
            g1_ok = False
        if row["case"] not in positive and row["parents"] >= 10000 and row["accept_rate"] > 0.0:
            g1_ok = False
    verdicts["G1_certificate_selectivity"] = g1_ok

    def _monotone(case_rows, field):
        by_parents = {row["parents"]: row[field] for row in case_rows}
        ordered = [by_parents[parents] for parents in sorted(by_parents)]
        return all(late < early for early, late in zip(ordered, ordered[1:]))

    g2_ok = True
    for (group, name) in {(row["group"], row["case"]) for row in rows}:
        case_rows = [
            row for row in rows if row["group"] == group and row["case"] == name and row["children"] == 4
        ]
        if len(case_rows) == len(PARENTS_GRID) and not _monotone(case_rows, "edge_error_median"):
            # Zero-operator cells sit at the noise floor where medians can
            # tie; only strictly-positive-signal cases gate monotonicity.
            if name in {"closed_chain", "full_orders", "truncated", "hallucinated_path", "isospectral_mismatch"}:
                g2_ok = False
    verdicts["G2_convergence_in_n"] = g2_ok

    gauge = [record["controls"]["gauge_invariance_max_change"] for key, record in records.items() if not key.startswith("C/")]
    verdicts["G3_gauge_invariance_max"] = max(gauge)
    verdicts["G3_gauge_ok"] = max(gauge) < 1e-8

    truncated_rows = [row for row in rows if row["case"] == "truncated" and row["parents"] == max(PARENTS_GRID)]
    analytic = (0.8 * 0.6) ** 3
    verdicts["G5_truncated_delta_op_analytic"] = analytic
    verdicts["G5_ok"] = all(
        row["delta_op_ci"][0] - 0.02 <= analytic <= row["delta_op_ci"][1] + 0.02 for row in truncated_rows
    )

    c_records = {key: records[key] for key in records if key.startswith("C/")}
    verdicts["G6_naive_product_min_error"] = min(
        record["naive_sigma_product_max_error"] for record in c_records.values()
    )
    verdicts["G6_closure_identity_error"] = c_records["C/nonnormal_markov"]["closure_identity_max_error"]
    verdicts["G6_ok"] = (
        verdicts["G6_naive_product_min_error"] > 0.2
        and verdicts["G6_closure_identity_error"] < 1e-10
    )

    output = {
        "certify_threshold": CERTIFY_THRESHOLD,
        "unit_count": len(records),
        "rows": rows,
        "verdicts": verdicts,
    }
    (root / "aggregate.json").write_text(json.dumps(output, indent=2))
    (root / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
