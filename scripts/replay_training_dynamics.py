"""E-B2 (approximate): edge mass against composed closure, along training.

The exact nilpotent-in-the-wild figure wants per-edge ||C_edge||_F and
sigma_1(C_comp) logged each epoch; the runs on disk logged
`train/edge_trace_sum` and `train/closure_ratio` instead.  Those two
still answer the qualitative question -- does the additive arm push edge
mass up while the composition fails to follow? -- so this replays them
from the CSV logs at zero cost, and the exact version waits on new
instrumentation.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "report_20260822"
OUT.mkdir(parents=True, exist_ok=True)


def read_curve(path, keys):
    rows = {key: [] for key in keys}
    epochs = {key: [] for key in keys}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            for key in keys:
                value = row.get(key, "")
                if value not in ("", None):
                    rows[key].append(float(value))
                    epochs[key].append(float(row["epoch"]))
    return epochs, rows


def main() -> None:
    arms = [("product_endpoint", "V7 composed", "#c0392b"),
            ("additive_mview", "additive (path-blind)", "#2980b9"),
            ("final_mview", "flat M-view", "#7f8c8d")]
    keys = ["train/edge_trace_sum", "train/closure_ratio"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4)) 
    for variant, label, color in arms:
        path = (ROOT / "results/gate1/gate1_20260820_v8/units" /
                f"{variant}__seed1" / "train_logs/version_0/metrics.csv")
        if not path.is_file():
            continue
        epochs, rows = read_curve(path, keys)
        for axis, key, title in zip(axes, keys,
                                    ("edge mass (sum of per-edge traces)",
                                     "closure ratio")):
            if rows[key]:
                axis.plot(epochs[key], rows[key], label=label, color=color, lw=1.6)
                axis.set_title(title, fontsize=11)
                axis.set_xlabel("epoch")
    axes[0].set_ylabel("train/edge_trace_sum")
    axes[1].set_ylabel("train/closure_ratio")
    axes[0].legend(fontsize=8)
    fig.suptitle("Edge mass rises in every arm; only the composed arm converts it into closure",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dynamics.png", dpi=150, bbox_inches="tight")

    print(f"{'arm':22s} {'final edge mass':>16s} {'final closure ratio':>20s}")
    for variant, label, _ in arms:
        path = (ROOT / "results/gate1/gate1_20260820_v8/units" /
                f"{variant}__seed1" / "train_logs/version_0/metrics.csv")
        if not path.is_file():
            continue
        _, rows = read_curve(path, keys)
        edge = rows["train/edge_trace_sum"][-1] if rows["train/edge_trace_sum"] else float("nan")
        ratio = rows["train/closure_ratio"][-1] if rows["train/closure_ratio"] else float("nan")
        print(f"{label:22s} {edge:>16.3f} {ratio:>20.3f}")
    print(f"\nwrote {OUT / 'fig_dynamics.png'}")


if __name__ == "__main__":
    main()
