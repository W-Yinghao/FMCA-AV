"""Apply the frozen layer-wise predictions (prereg 2026-08-23).

Reads layerwise_profile.json records and evaluates P1-P5 exactly as
frozen, including the evidence standard: separation is claimed only
when the three-seed ranges do not overlap; anything weaker is a trend.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (1, 2, 3)
INTERIOR, ENDPOINT = 2, 3
SETS = [("CIFAR-10", "results/gate1/gate1_20260820_v8/units"),
        ("CIFAR-100", "results/gate1/gate1_20260821_c100pilot/units")]


def load(root, variant, seed):
    path = ROOT / root / f"{variant}__seed{seed}" / "layerwise_profile.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    return record if record.get("status") == "complete" else None


def series(root, variant, key):
    """Per-seed vectors of a per-stage quantity."""

    out = []
    for seed in SEEDS:
        record = load(root, variant, seed)
        if record is None:
            continue
        if key == "probe":
            out.append([s["probe"]["test_accuracy"] * 100 for s in record["stages"]])
        elif key == "rank":
            out.append([s["spectrum"]["effective_rank"] for s in record["stages"]])
        elif key == "cca":
            out.append([i["canonical_correlations"]["mean_top_k"]
                        for i in record["interfaces"]])
    return np.array(out) if out else None


def disjoint(a, b):
    """Frozen evidence standard: do the two seed ranges fail to overlap?"""

    return min(a) > max(b) or min(b) > max(a)


def contrast_block(name, root, treatment, control, key, label):
    t, c = series(root, treatment, key), series(root, control, key)
    if t is None or c is None or len(t) != len(c):
        print(f"  {name}: incomplete ({0 if t is None else len(t)} vs "
              f"{0 if c is None else len(c)} seeds)")
        return None
    delta = t - c
    print(f"  {name} ({label})")
    for stage in range(delta.shape[1]):
        tag = {INTERIOR: "  <- interior interface", ENDPOINT: "  <- endpoint"}.get(stage, "")
        print(f"    stage {stage}: D = {delta[:, stage].mean():+7.2f}   "
              f"per seed " + " ".join(f"{x:+6.2f}" for x in delta[:, stage]) + tag)
    return delta


def main() -> None:
    for title, root in SETS:
        print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")

        for key, label in (("probe", "probe accuracy, points"),
                           ("rank", "effective rank")):
            print(f"\n-- {label} --")
            for variant in ("product_endpoint", "additive_mview", "final_mview"):
                values = series(root, variant, key)
                if values is None:
                    continue
                print(f"  {variant:18s} " + "  ".join(
                    f"s{i}: {values[:, i].mean():6.2f}" for i in range(values.shape[1])))

        print("\n-- preregistered contrasts --")
        v7_probe = contrast_block("V7 - flat", root, "product_endpoint", "final_mview",
                                  "probe", "probe points")
        add_probe = contrast_block("additive - flat", root, "additive_mview", "final_mview",
                                   "probe", "probe points")
        v7_rank = contrast_block("V7 - flat", root, "product_endpoint", "final_mview",
                                 "rank", "effective rank")
        add_rank = contrast_block("additive - flat", root, "additive_mview", "final_mview",
                                  "rank", "effective rank")

        print("\n-- verdicts --")
        if v7_probe is not None:
            interior, endpoint = v7_probe[:, INTERIOR], v7_probe[:, ENDPOINT]
            passed = interior.mean() > endpoint.mean()
            clean = disjoint(interior, endpoint)
            print(f"  P1 interior redistribution: D(2)={interior.mean():+.2f} vs "
                  f"D(3)={endpoint.mean():+.2f} -> "
                  f"{'PASS' if passed else 'REFUTED'}"
                  f"{' (seed ranges disjoint)' if clean else ' (ranges overlap: trend only)'}")
            print(f"     strong form D(2)>0: "
                  f"{'yes' if interior.min() > 0 else 'no'} "
                  f"(min per seed {interior.min():+.2f})")
        if v7_rank is not None:
            interior, endpoint = v7_rank[:, INTERIOR], v7_rank[:, ENDPOINT]
            print(f"  P2 anti-compression: D_rank(2)={interior.mean():+.2f} "
                  f"({'>' if interior.mean() > 0 else '<='}0), "
                  f"vs D_rank(3)={endpoint.mean():+.2f} -> "
                  f"{'PASS' if interior.mean() > 0 and interior.mean() > endpoint.mean() else 'REFUTED'}")
        for tag, v7_delta, add_delta in (("probe", v7_probe, add_probe),
                                         ("rank", v7_rank, add_rank)):
            if v7_delta is not None and add_delta is not None:
                a, b = v7_delta[:, INTERIOR], add_delta[:, INTERIOR]
                print(f"  P4 mechanism specificity ({tag}): V7 {a.mean():+.2f} vs "
                      f"additive {b.mean():+.2f} at the interior -> "
                      f"{'specific' if disjoint(a, b) and a.mean() > b.mean() else 'NOT specific to composition'}")

        print("\n-- P3 interface retention (mean top-8 canonical correlation) --")
        for variant in ("product_endpoint", "additive_mview", "final_mview"):
            values = series(root, variant, "cca")
            if values is not None:
                print(f"  {variant:18s} " + "  ".join(
                    f"{i}->{i+1}: {values[:, i].mean():.3f}" for i in range(values.shape[1])))

    print("\nEvidence standard (frozen): separation claimed only on disjoint")
    print("three-seed ranges; overlapping ranges are reported as trends.")


if __name__ == "__main__":
    main()
