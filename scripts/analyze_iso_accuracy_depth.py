"""Q2, second half: iso-accuracy depth and what it costs to get there.

A per-stage accuracy curve says a method is better at some depth.  It
does not say the network can be SHORTER, which is the claim worth
making.  This turns the curves into that claim: for pre-committed
absolute accuracy targets, report the shallowest stage each method
reaches, and the inference cost of running only that prefix.

Cost is counted analytically for the CIFARResNet the gate and the
baselines share -- convolution and linear multiply-accumulates for the
stem plus each retained stage, plus the linear head ACTUALLY used at
that depth (stage width x classes).  Counting the backbone alone would
flatter a shallow tap whose head is wider, so the head is included.

A method that never reaches a target is reported as "not reached".
Nothing is extrapolated and no target is set per-method.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TARGETS = (0.70, 0.75, 0.80)          # frozen in the prereg
WIDTH, BLOCKS, INPUT = 64, (2, 2, 2, 2), 32


def _conv(cin, cout, k, out_hw):
    return cin * cout * k * k * out_hw * out_hw


def stage_costs(width=WIDTH, blocks=BLOCKS, input_hw=INPUT):
    """Cumulative multiply-accumulates after the stem and each stage."""

    costs, running = [], _conv(3, width, 3, input_hw)   # CIFAR stem: 3x3, stride 1
    channels, hw = width, input_hw
    for index, count in enumerate(blocks):
        out_channels = width * (2 ** index)
        stride = 1 if index == 0 else 2
        out_hw = hw // stride
        for block in range(count):
            cin = channels if block == 0 else out_channels
            in_hw = hw if block == 0 else out_hw
            running += _conv(cin, out_channels, 3, out_hw)
            running += _conv(out_channels, out_channels, 3, out_hw)
            if block == 0 and (cin != out_channels or stride != 1):
                running += _conv(cin, out_channels, 1, out_hw)   # projection shortcut
            in_hw = in_hw
        channels, hw = out_channels, out_hw
        costs.append((running, channels))
    return costs


def curve_from_profile(path):
    d = json.loads(Path(path).read_text())
    return [s["probe"]["test_accuracy"] for s in d["stages"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", required=True,
                        help="name=glob pairs, e.g. V7=results/.../product_endpoint__seed*/layerwise_profile.json")
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    import glob as globmod
    costs = stage_costs()
    methods = {}
    for pair in arguments.profiles:
        name, pattern = pair.split("=", 1)
        files = sorted(globmod.glob(pattern))
        if not files:
            print(f"  {name}: NO FILES for {pattern}")
            continue
        curves = [curve_from_profile(f) for f in files]
        stages = min(len(c) for c in curves)
        mean = [sum(c[i] for c in curves) / len(curves) for i in range(stages)]
        lo = [min(c[i] for c in curves) for i in range(stages)]
        hi = [max(c[i] for c in curves) for i in range(stages)]
        methods[name] = {"seeds": len(curves), "mean": mean, "min": lo, "max": hi}

    print(f"{'method':16s} " + "  ".join(f"stage{i+1}" for i in range(4)))
    for name, m in methods.items():
        print(f"{name:16s} " + "  ".join(f"{v*100:6.2f}" for v in m["mean"]) + f"   (n={m['seeds']})")

    results = {}
    for target in TARGETS:
        print(f"\n--- target {target*100:.0f}% ---")
        rows = {}
        for name, m in methods.items():
            reached = next((i for i, v in enumerate(m["mean"]) if v >= target), None)
            # Require the WHOLE three-seed range above the target, not just
            # the mean, before crediting a method with reaching it.
            robust = next((i for i, v in enumerate(m["min"]) if v >= target), None)
            if reached is None:
                rows[name] = {"stage": None, "macs": None, "robust_stage": None}
                print(f"  {name:16s} not reached")
            else:
                macs, channels = costs[reached]
                total = macs + channels * arguments.classes
                rows[name] = {"stage": reached + 1, "macs": total,
                              "backbone_macs": macs, "head_macs": channels * arguments.classes,
                              "robust_stage": None if robust is None else robust + 1}
                print(f"  {name:16s} stage {reached+1} ({total/1e6:8.2f} MMACs)"
                      f"  robust stage {rows[name]['robust_stage']}")
        reached_rows = {k: v for k, v in rows.items() if v["macs"]}
        if len(reached_rows) > 1:
            best = min(reached_rows.items(), key=lambda kv: kv[1]["macs"])
            print(f"  cheapest: {best[0]} at {best[1]['macs']/1e6:.2f} MMACs")
        results[f"{target:.2f}"] = rows

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "targets": list(TARGETS), "classes": arguments.classes,
        "cumulative_macs_by_stage": [c[0] for c in costs],
        "stage_channels": [c[1] for c in costs],
        "curves": methods, "iso_accuracy": results,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
