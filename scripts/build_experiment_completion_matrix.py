#!/usr/bin/env python3
"""Audit E0--E10 completion against concrete result assets and formal states."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


VERSION = SCIENTIFIC_CORRECTNESS_VERSION
RESULTS_ROOT = Path("results/postfix") / VERSION
STATE_ROOT = Path("results/orchestration")


def build_requirements() -> dict[str, tuple[list[str], list[str]]]:
    result = lambda group, name: str(RESULTS_ROOT / group / name)
    state = lambda name: str(STATE_ROOT / name)
    return {
        "E0": ([result("e1", "exact_discrete_channels.csv"), result("e1", "gaussian_recovery.csv")], []),
        "E1": ([result("e1", "finite_sample_recovery.csv"), result("e1", "nonlinear_recovery.csv"), result("e1", "estimator_baselines.csv")], []),
        "E2": ([result("e2", "gradient_variance_table.csv")], []),
        "E3": ([result("e3", "cifar_numerics_table.csv"), result("e3", "estimator_controls_summary.csv")], [state(f"e3_imagenet100_recheck_{VERSION}.json")]),
        "E4": ([result("e4", "aggregation_ablation.csv")], []),
        "E5": ([result("e5", "matched_ssl_runs.csv"), result("e5", "matched_compute_runs.csv")], [state("formal_ssl_postfix_state.json"), state(f"matched_compute_{VERSION}.json"), state(f"imagenet_formal_{VERSION}.json")]),
        "E6": ([result("e6", "generalization_transfer_table.csv"), result("e6", "robustness_table.csv"), result("e6", "robustness_per_corruption.csv")], [state(f"formal_transfer_{VERSION}.json"), state(f"formal_low_label_{VERSION}.json"), state(f"formal_imagenet_low_label_{VERSION}.json")]),
        "E7": ([result("e7", "factor_probe_summary.csv"), result("e7", "tsd_utility_table.csv"), result("e7", "tsd_calibration_summary.csv"), result("e7", "tsd_data_processing_chain.csv"), result("e7", "image_data_processing_chain.csv")], [state(f"factor_suite_{VERSION}.json")]),
        "E8": ([result("e8", "exact_markov_conditions.csv"), result("e8", "continuous_markov_conditions.csv")], [state(f"e8_{VERSION}.json")]),
        "E9": ([result("e9", "localization_table.csv"), result("e9", "localization_summary.csv"), result("e9", "cnn_composition_table.csv")], [state(f"formal_localization_{VERSION}.json")]),
        "E10": ([result("e10", "complexity_table.csv"), result("e10", "operator_complexity_table.csv"), result("e10", "flops_profile_table.csv"), result("e10", "ddp_scaling_table.csv")], [state(f"e10_{VERSION}.json")]),
    }


def nonempty(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix == ".csv":
        with path.open(encoding="utf-8") as handle:
            return sum(1 for _ in handle) > 1
    return True


def main() -> int:
    requirements = build_requirements()
    blockers_path = Path("results/orchestration/external_blockers.json")
    blockers = json.loads(blockers_path.read_text(encoding="utf-8")) if blockers_path.is_file() else {}
    records = []
    for experiment, (files, states) in requirements.items():
        missing_files = [value for value in files if not nonempty(Path(value))]
        state_values = {}
        for value in states:
            path = Path(value)
            if not path.is_file():
                state_values[value] = "MISSING"
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            state_values[value] = (
                str(payload.get("state", "MISSING"))
                if payload.get("scientific_correctness_version") == VERSION
                else "VERSION_MISMATCH"
            )
        bad_states = {key: value for key, value in state_values.items() if value != "SUCCEEDED"}
        blocker = blockers.get(experiment, "") if isinstance(blockers, dict) else ""
        status = "BLOCKED_EXTERNAL" if blocker else "COMPLETED" if not missing_files and not bad_states else "INCOMPLETE"
        records.append({
            "scientific_correctness_version": VERSION,
            "experiment": experiment, "status": status, "required_files": ";".join(files),
            "missing_or_empty_files": ";".join(missing_files), "required_states": json.dumps(state_values, sort_keys=True),
            "external_blocker": blocker,
        })
    output = RESULTS_ROOT / "index"; output.mkdir(parents=True, exist_ok=True)
    fields = ["scientific_correctness_version", "experiment", "status", "required_files", "missing_or_empty_files", "required_states", "external_blocker"]
    temporary = output / "experiment_completion_matrix.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(records)
    temporary.replace(output / "experiment_completion_matrix.csv")
    temporary = output / "experiment_completion_matrix.json.tmp"
    temporary.write_text(json.dumps({"scientific_correctness_version": VERSION, "experiments": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output / "experiment_completion_matrix.json")
    incomplete = [record["experiment"] for record in records if record["status"] == "INCOMPLETE"]
    print(json.dumps({"completed": sum(record["status"] == "COMPLETED" for record in records), "blocked_external": sum(record["status"] == "BLOCKED_EXTERNAL" for record in records), "incomplete": incomplete}, indent=2))
    if incomplete:
        raise RuntimeError("experiment completion audit is incomplete: " + ",".join(incomplete))
    return 0


if __name__ == "__main__": raise SystemExit(main())
