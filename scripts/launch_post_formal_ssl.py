#!/usr/bin/env python3
"""Run post-SSL matched-compute and low-label tracks in one CPU watcher."""

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION

from scripts.launch_matched_compute_probes import main as matched_compute_main
from scripts.launch_formal_low_label import main as low_label_main
from scripts.launch_e3_imagenet100_recheck import main as imagenet100_numerics_main


def main() -> int:
    formal_state = "results/orchestration/formal_ssl_postfix_state.json"
    matched_state = f"results/orchestration/matched_compute_{SCIENTIFIC_CORRECTNESS_VERSION}.json"
    low_label_state = f"results/orchestration/formal_low_label_{SCIENTIFIC_CORRECTNESS_VERSION}.json"
    result = int(matched_compute_main([
        "--formal-state", formal_state, "--state-file", matched_state,
    ]))
    if result:
        return result
    result = int(low_label_main([
        "--formal-state", formal_state, "--state-file", low_label_state,
    ]))
    return result if result else int(imagenet100_numerics_main())


if __name__ == "__main__":
    raise SystemExit(main())
