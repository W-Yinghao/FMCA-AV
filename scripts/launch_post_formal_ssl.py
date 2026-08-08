#!/usr/bin/env python3
"""Run post-SSL matched-compute and low-label tracks in one CPU watcher."""

from scripts.launch_matched_compute_probes import main as matched_compute_main
from scripts.launch_formal_low_label import main as low_label_main
from scripts.launch_e3_imagenet100_recheck import main as imagenet100_numerics_main


def main() -> int:
    result = int(matched_compute_main())
    if result:
        return result
    result = int(low_label_main())
    return result if result else int(imagenet100_numerics_main())


if __name__ == "__main__":
    raise SystemExit(main())
