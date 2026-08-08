#!/usr/bin/env python3
"""Audit E0--E10 completion against concrete result assets and formal states."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REQUIREMENTS = {
    "E0": (["results/e1/exact_discrete_channels.csv", "results/e1/gaussian_recovery.csv"], []),
    "E1": (["results/e1/finite_sample_recovery.csv", "results/e1/nonlinear_recovery.csv", "results/e1/estimator_baselines.csv"], []),
    "E2": (["results/e2/gradient_variance_table.csv"], []),
    "E3": (["results/e3/cifar_numerics_table.csv", "results/e3/estimator_controls_summary.csv"], ["results/orchestration/e3_imagenet100_recheck_state.json"]),
    "E4": (["results/e4/aggregation_ablation.csv"], []),
    "E5": (["results/e5/matched_ssl_runs.csv", "results/e5/matched_compute_runs.csv"], ["results/orchestration/formal_ssl_state.json", "results/orchestration/imagenet_formal_state.json"]),
    "E6": (["results/e6/generalization_transfer_table.csv", "results/e6/robustness_table.csv", "results/e6/robustness_per_corruption.csv"], ["results/orchestration/formal_transfer_state.json", "results/orchestration/formal_low_label_state.json", "results/orchestration/formal_imagenet_low_label_state.json"]),
    "E7": (["results/e7/factor_probe_summary.csv", "results/e7/tsd_utility_table.csv", "results/e7/tsd_calibration_summary.csv", "results/e7/tsd_data_processing_chain.csv", "results/e7/image_data_processing_chain.csv"], ["results/orchestration/full_factor_probes_state.json"]),
    "E8": (["results/e8/exact_markov_conditions.csv", "results/e8/continuous_markov_conditions.csv"], []),
    "E9": (["results/e9/localization_table.csv", "results/e9/localization_summary.csv", "results/e9/cnn_composition_table.csv"], ["results/orchestration/formal_localization_state.json"]),
    "E10": (["results/e10/complexity_table.csv", "results/e10/operator_complexity_table.csv", "results/e10/flops_profile_table.csv", "results/e10/ddp_scaling_table.csv"], []),
}


def nonempty(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix == ".csv":
        with path.open(encoding="utf-8") as handle:
            return sum(1 for _ in handle) > 1
    return True


def main() -> int:
    blockers_path = Path("results/orchestration/external_blockers.json")
    blockers = json.loads(blockers_path.read_text(encoding="utf-8")) if blockers_path.is_file() else {}
    records = []
    for experiment, (files, states) in REQUIREMENTS.items():
        missing_files = [value for value in files if not nonempty(Path(value))]
        state_values = {}
        for value in states:
            path = Path(value)
            state_values[value] = (str(json.loads(path.read_text(encoding="utf-8")).get("state", "MISSING"))
                                   if path.is_file() else "MISSING")
        bad_states = {key: value for key, value in state_values.items() if value != "SUCCEEDED"}
        blocker = blockers.get(experiment, "") if isinstance(blockers, dict) else ""
        status = "BLOCKED_EXTERNAL" if blocker else "COMPLETED" if not missing_files and not bad_states else "INCOMPLETE"
        records.append({
            "experiment": experiment, "status": status, "required_files": ";".join(files),
            "missing_or_empty_files": ";".join(missing_files), "required_states": json.dumps(state_values, sort_keys=True),
            "external_blocker": blocker,
        })
    output = Path("results/index"); output.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "status", "required_files", "missing_or_empty_files", "required_states", "external_blocker"]
    temporary = output / "experiment_completion_matrix.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    temporary.replace(output / "experiment_completion_matrix.csv")
    temporary = output / "experiment_completion_matrix.json.tmp"
    temporary.write_text(json.dumps({"experiments": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output / "experiment_completion_matrix.json")
    incomplete = [record["experiment"] for record in records if record["status"] == "INCOMPLETE"]
    print(json.dumps({"completed": sum(record["status"] == "COMPLETED" for record in records), "blocked_external": sum(record["status"] == "BLOCKED_EXTERNAL" for record in records), "incomplete": incomplete}, indent=2))
    if incomplete:
        raise RuntimeError("experiment completion audit is incomplete: " + ",".join(incomplete))
    return 0


if __name__ == "__main__": raise SystemExit(main())
