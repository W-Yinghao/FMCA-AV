"""Emit the meeting results tables straight from the result JSONs.

CPU-only, re-runnable: every number in meeting_20260823/RESULTS_TABLES.md
comes from a unit.json / remeasure / spectroscopy file, never typed by hand.
Usage: make_meeting_tables.py [output_path]
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
R = Path("results")
lines = []
W = lines.append


def stats(paths):
    a, d, k = [], [], []
    for p in paths:
        if not os.path.isfile(p):
            continue
        u = json.load(open(p))
        if u.get("status") != "complete":
            continue
        a.append(u["linear_probe"]["test_accuracy"] * 100)
        d.append(u["certificate"]["normalized_closure_defect"])
        k.append(u["knn_accuracy"] * 100)
    return (np.array(a), np.array(d), np.array(k)) if a else None


W("# 结果表（数字全部由 result JSON 直出）\n")
W("由 `scripts/make_meeting_tables.py` 生成；任何时刻重跑即刷新。\n")

W("\n## 表 1 — Gate v8 主表（CIFAR-10，全保真树，200 epochs）\n")
W("| 变体 | n | linear probe (%) | kNN (%) | 闭合亏差 |")
W("|---|---|---|---|---|")
V8 = "results/gate1/gate1_20260820_v8/units"
for v, label in [("final_2view", "V1 flat 2-view"), ("final_mview", "V2 flat M-view (锚)"),
                 ("additive_2view", "V3 additive 2-view"), ("additive_mview", "V4 additive M-view"),
                 ("amdim_cross", "V5 AMDIM-cross"), ("product_only", "V6 product-only"),
                 ("product_endpoint", "**V7 full method**")]:
    s = stats([f"{V8}/{v}__seed{i}/unit.json" for i in range(1, 6)])
    if s:
        a, d, k = s
        W(f"| {label} | {len(a)} | {a.mean():.2f} ± {a.std():.2f} | {k.mean():.2f} | {d.mean():.3f} ± {d.std():.3f} |")

W("\n## 表 2 — CIFAR-100（仪器出现分辨力的设定）\n")
W("| 变体 | n | linear probe (%) | 闭合亏差 | 每种子亏差 |")
W("|---|---|---|---|---|")
C = "results/gate1/gate1_20260821_c100pilot/units"
for v, label in [("final_mview", "flat M-view"), ("product_endpoint", "**V7 full method**"),
                 ("additive_mview", "additive M-view"), ("final_2view", "flat 2-view"),
                 ("additive_2view", "additive 2-view"), ("amdim_cross", "AMDIM-cross"),
                 ("product_only", "product-only")]:
    s = stats([f"{C}/{v}__seed{i}/unit.json" for i in (1, 2, 3)])
    if s:
        a, d, k = s
        each = ", ".join(f"{x:.3f}" for x in d)
        W(f"| {label} | {len(a)} | {a.mean():.2f} ± {a.std():.2f} | {d.mean():.3f} ± {d.std():.3f} | {each} |")

W("\n## 表 3 — Plug-in 研究（闭合正则挂到外部 SSL 基座，CIFAR-10）\n")
W("| 方法 | 行 | n | linear probe (%) | 闭合亏差 |")
W("|---|---|---|---|---|")
P = "results/plugin/plugin_20260820_v1/units"
for base in ("barlow_twins", "vicreg", "frossl"):
    for row, label in [(f"{base}_base", "base"), (f"{base}_plugin_b16", "+plugin β=16"),
                       (f"{base}_plugin", "+plugin β=32"), (f"{base}_plugin_b48", "+plugin β=48")]:
        s = stats(sorted(glob.glob(f"{P}/{row}__seed*/unit.json")))
        if s:
            a, d, k = s
            note = "  ⚠️ 基座在统一配方下不稳定" if base == "frossl" and label == "base" else ""
            W(f"| {base} | {label} | {len(a)} | {a.mean():.2f} ± {a.std():.2f} | {d.mean():.3f} ± {d.std():.3f} |{note}")

W("\n## 表 4 — 消融与负控（CIFAR-10，v8 配方）\n")
W("| 单元 | n | linear probe (%) | 闭合亏差 | 结论 |")
W("|---|---|---|---|---|")
for pat, label, n, note in [
        ("results/gate1/gate1_20260820_v8/units/product_endpoint__seed{}/unit.json", "V7 参照 (nested)", 5, "基准"),
        ("results/gate1/gate1_20260821_v8_parallel/units/product_endpoint__seed{}/unit.json", "star/parallel 树负控", 3, "**零结果**：与嵌套无法区分"),
        ("results/gate1/gate1_20260822_c100_parallel/units/product_endpoint__seed{}/unit.json", "star 树负控 @ CIFAR-100", 3, "在有分辨力的设定重做"),
        ("results/gate1/gate1_20260821_v8_alpha0/units/product_endpoint__seed{}/unit.json", "α=0（无边 bootstrap）", 1, "EMA 目标独立治愈冷启动"),
        ("results/gate1/gate1_20260821_v8_m4/units/product_endpoint__seed{}/unit.json", "M=4 端点", 1, "端点视图数重要")]:
    s = stats([pat.format(i) for i in range(1, n + 1)])
    if s:
        a, d, k = s
        W(f"| {label} | {len(a)} | {a.mean():.2f} ± {a.std():.2f} | {d.mean():.3f} ± {d.std():.3f} | {note} |")

W("\n## 表 5 — 仪器研究：Stage-B 校准 2500 → 5000（合池 val，ridge 0.1）\n")
W("| 行 | 校准 2500 | 校准 5000 | 变化 |")
W("|---|---|---|---|")
for v, label in [("product_endpoint", "V7 显式闭合"), ("additive_2view", "V3 加性"),
                 ("final_mview", "flat M-view"), ("final_2view", "flat 2-view")]:
    b, p = [], []
    for s_ in (1, 2, 3):
        f1 = f"{V8}/{v}__seed{s_}/remeasure_ridge.json"
        f2 = f"{V8}/{v}__seed{s_}/remeasure_ridge_pooled.json"
        if os.path.isfile(f1):
            b.append(json.load(open(f1))["0.1"]["normalized_closure_defect"])
        if os.path.isfile(f2):
            p.append(json.load(open(f2))["0.1"]["normalized_closure_defect"])
    if b and p:
        W(f"| {label} | {np.mean(b):.3f} (n={len(b)}) | {np.mean(p):.3f} (n={len(p)}) | {np.mean(p) - np.mean(b):+.3f} |")

W("\n## 表 6 — 探测深度画像（stage 级组合质量保留率）\n")
W("| backbone | stage1 | stage2 | stage3 | stage4 | 最佳探测层 |")
W("|---|---|---|---|---|---|")
for f, label in [("resnet50_IMAGENET1K_V2_stage.json", "resnet50 预训练"),
                 ("resnet50_none_stage.json", "resnet50 随机初始化"),
                 ("resnet152_IMAGENET1K_V2_cifar10_stage.json", "resnet152 预训练"),
                 ("resnet18_IMAGENET1K_V1_cifar10_stage.json", "resnet18 预训练"),
                 ("resnet50_IMAGENET1K_V2_cifar100_stage.json", "resnet50 预训练 / CIFAR-100")]:
    p = R / "probing_depth" / f
    if p.is_file():
        d = json.loads(p.read_text())
        r = [f"{c['retention']:.3f}" for c in d["certificate_curve"]]
        W(f"| {label} | {' | '.join(r)} | stage {d['best_probe_layer']} |")

W("\n## 表 7 — 通道谱学（扰动单边，观察各边算子范数）\n")
W("| 扫描 | edge0 ‖·‖_F | edge1 ‖·‖_F | 判读 |")
W("|---|---|---|---|")
for f, label, verdict in [("v8_edge1_min_scale.json", "edge1 min_scale 0.2→0.95", "**只动 edge1**"),
                          ("v8_edge0_flip_probability.json", "edge0 flip 0→1", "零响应（空通道）"),
                          ("edge1_grayscale_probability.json", "edge1 grayscale", "只动 edge1"),
                          ("edge0_color_jitter_strength.json", "edge0 color jitter", "动 edge0")]:
    p = R / "spectroscopy" / f
    if p.is_file():
        d = json.loads(p.read_text())
        vals = sorted(d["sweep"], key=float)
        e0 = [d["sweep"][v]["edge_frobenius_norms"][0] for v in vals]
        e1 = [d["sweep"][v]["edge_frobenius_norms"][1] for v in vals]
        W(f"| {label} | {min(e0):.2f} → {max(e0):.2f} | {min(e1):.2f} → {max(e1):.2f} | {verdict} |")

out = Path(sys.argv[1] if len(sys.argv) > 1 else "meeting_20260823/RESULTS_TABLES.md")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print(f"wrote {out} ({len(lines)} lines)")
