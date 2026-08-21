"""Generate presentation figures from committed result JSONs.

CPU-only; safe to re-run any time (skips whatever is not on disk yet).
Writes PNGs into outputs/report_20260822/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "report_20260822"
OUT.mkdir(parents=True, exist_ok=True)
V8 = ROOT / "results/gate1/gate1_20260820_v8/units"

VARIANT_LABELS = [
    ("final_2view", "V1 flat 2-view"),
    ("final_mview", "V2 flat M-view"),
    ("additive_2view", "V3 additive 2-view"),
    ("additive_mview", "V4 additive M-view"),
    ("amdim_cross", "V5 AMDIM-cross"),
    ("product_only", "V6 product-only"),
    ("product_endpoint", "V7 full method"),
]


def load_unit(variant, seed):
    p = V8 / f"{variant}__seed{seed}" / "unit.json"
    if p.is_file():
        d = json.loads(p.read_text())
        if d.get("status") == "complete":
            return d
    return None


def fig_arc():
    milestones = [
        ("v4\nraw recipe", 48.3), ("v6\nβ=128 rescale", 72.4),
        ("v7\nM=8 + leaf", 75.4), ("v7\nEMA target", 82.8),
        ("v8\nfull-fidelity tree", 85.4),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = range(len(milestones))
    ys = [m[1] for m in milestones]
    ax.plot(xs, ys, "o-", lw=2, ms=8, color="#c0392b")
    ax.axhline(89.0, ls="--", color="gray", lw=1)
    ax.text(0.05, 89.4, "flat anchor 89.0", color="gray", fontsize=9)
    for x, (label, y) in zip(xs, milestones):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, weight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([m[0] for m in milestones], fontsize=9)
    ax.set_ylabel("linear probe accuracy (%)")
    ax.set_title("V7 full-method optimization arc (each step single-variable)")
    ax.set_ylim(40, 95)
    fig.tight_layout()
    fig.savefig(OUT / "fig_arc.png", dpi=150)


def fig_v8_main():
    labels, means, spreads, colors = [], [], [], []
    for variant, label in VARIANT_LABELS:
        accs = [u["linear_probe"]["test_accuracy"] * 100
                for s in range(1, 6) if (u := load_unit(variant, s))]
        if not accs:
            continue
        labels.append(f"{label}\n(n={len(accs)})")
        means.append(np.mean(accs))
        spreads.append(np.std(accs))
        colors.append("#c0392b" if variant == "product_endpoint"
                      else "#7f8c8d" if "final" in variant else "#2980b9")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=spreads, capsize=4, color=colors, alpha=0.85)
    for x, m in zip(xs, means):
        ax.text(x, m + 1.2, f"{m:.1f}", ha="center", fontsize=9, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("linear probe accuracy (%)")
    ax.set_title("Gate v8 (CIFAR-10, full-fidelity tree, 200 epochs)")
    ax.set_ylim(40, 96)
    fig.tight_layout()
    fig.savefig(OUT / "fig_v8_main.png", dpi=150)


def fig_instrument():
    rows = ["product_endpoint", "additive_2view", "final_mview",
            "final_2view", "product_only"]
    names = ["V7 explicit", "V3 additive", "V2 flat M", "V1 flat 2", "V6 product-only"]
    base_vals, pooled_vals, keep = [], [], []
    for row, name in zip(rows, names):
        b, p = [], []
        for s in (1, 2, 3):
            unit = V8 / f"{row}__seed{s}"
            f = unit / "remeasure_ridge.json"
            g = unit / "remeasure_ridge_pooled.json"
            if f.is_file():
                b.append(json.loads(f.read_text())["0.1"]["normalized_closure_defect"])
            elif (u := load_unit(row, s)):
                b.append(u["certificate"]["normalized_closure_defect"])
            if g.is_file():
                p.append(json.loads(g.read_text())["0.1"]["normalized_closure_defect"])
        if b:
            keep.append(name)
            base_vals.append(np.mean(b))
            pooled_vals.append(np.mean(p) if p else np.nan)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    xs = np.arange(len(keep))
    ax.bar(xs - 0.18, base_vals, 0.36, label="calibration 2500", color="#7f8c8d")
    ax.bar(xs + 0.18, pooled_vals, 0.36, label="calibration 5000 (pooled)", color="#27ae60")
    ax.set_xticks(xs)
    ax.set_xticklabels(keep, fontsize=9)
    ax.set_ylabel("normalized closure defect (ridge 0.1)")
    ax.set_title("Instrument study: doubling Stage-B calibration")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_instrument.png", dpi=150)


def fig_spectroscopy():
    files = sorted((ROOT / "results/spectroscopy").glob("*.json"))
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(4.2 * len(files), 3.6))
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, files):
        d = json.loads(f.read_text())
        values = sorted(d["sweep"], key=float)
        e0 = [d["sweep"][v]["edge_frobenius_norms"][0] for v in values]
        e1 = [d["sweep"][v]["edge_frobenius_norms"][1] for v in values]
        ax.plot([float(v) for v in values], e0, "o-", label="edge 0 (root→mid)")
        ax.plot([float(v) for v in values], e1, "s-", label="edge 1 (mid→leaf)")
        ax.set_xlabel(f"edge {d['edge']} {d['parameter']}", fontsize=9)
        ax.set_ylabel("edge operator ‖·‖_F")
        ax.legend(fontsize=8)
        ax.set_title(f.stem, fontsize=9)
    fig.suptitle("Channel spectroscopy: perturbing one edge moves only that edge's operator", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_spectroscopy.png", dpi=150, bbox_inches="tight")


def fig_depth():
    curves = {
        "resnet50 pretrained": "resnet50_IMAGENET1K_V2_stage.json",
        "resnet50 random init": "resnet50_none_stage.json",
        "resnet152 pretrained": "resnet152_IMAGENET1K_V2_cifar10_stage.json",
        "resnet18 pretrained": "resnet18_IMAGENET1K_V1_cifar10_stage.json",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for label, name in curves.items():
        p = ROOT / "results/probing_depth" / name
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        prefixes = [c["prefix"] for c in d["certificate_curve"]]
        ax1.plot(prefixes, [c["retention"] for c in d["certificate_curve"]],
                 "o-", label=label)
        ax2.plot(range(len(d["layer_probe_accuracy"])),
                 [a * 100 for a in d["layer_probe_accuracy"]], "o-", label=label)
    ax1.set_xlabel("stage prefix"), ax1.set_ylabel("composed-mass retention")
    ax1.set_title("certificate retention per stage"), ax1.legend(fontsize=8)
    ax2.set_xlabel("stage"), ax2.set_ylabel("layer probe accuracy (%)")
    ax2.set_title("ground truth: layerwise probes"), ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_depth.png", dpi=150)


def fig_plugin():
    units = ROOT / "results/plugin/plugin_20260820_v1/units"
    bases = {}
    for p in units.glob("*__seed*/unit.json"):
        d = json.loads(p.read_text())
        if d.get("status") != "complete" or d.get("row_tag"):
            continue
        name = d["base"]
        if p.parent.name.split("__")[0] not in (f"{name}_base", f"{name}_plugin"):
            continue  # exclude beta-sweep rows from the headline panel
        kind = "plugin" if d["plugin_enabled"] else "base"
        bases.setdefault(name, {}).setdefault(kind, []).append(
            (d["linear_probe"]["test_accuracy"] * 100,
             d["certificate"]["normalized_closure_defect"]))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, kinds in sorted(bases.items()):
        if "base" not in kinds or "plugin" not in kinds:
            continue
        b = np.mean(kinds["base"], axis=0)
        g = np.mean(kinds["plugin"], axis=0)
        ax.annotate("", xy=(g[1], g[0]), xytext=(b[1], b[0]),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#8e44ad"))
        ax.scatter([b[1]], [b[0]], s=70, color="#7f8c8d", zorder=3)
        ax.scatter([g[1]], [g[0]], s=70, color="#8e44ad", zorder=3)
        flag = " (base unstable)" if name == "frossl" else ""
        ax.text(b[1] + 0.008, b[0], f"{name}{flag}", fontsize=9)
    ax.set_xlabel("normalized closure defect")
    ax.set_ylabel("linear probe accuracy (%)")
    ax.set_title("Plug-in study: closure regularizer on external SSL bases\n(gray = base, purple = +plugin; arrows show the exchange)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_plugin.png", dpi=150)


for fn in (fig_arc, fig_v8_main, fig_instrument, fig_spectroscopy, fig_depth, fig_plugin):
    try:
        fn()
        print(fn.__name__, "written")
    except Exception as error:  # keep going: partial figures still useful
        print(fn.__name__, "SKIPPED:", repr(error))
