"""Probing-depth study v1: certificate-vs-depth curves on a pretrained model.

For a pretrained backbone (torchvision resnet18/50/152, random init as a
control, or a repo checkpoint), this computes at BLOCK resolution:
  - per-interface adjacent operators (view A block l vs conditional mean
    of M strong views' block l+1) in frozen Stage-B coordinates,
  - depth-prefix triplets: C_comp(0->l) vs C_dir(0->l) with the Weyl
    certificate s_cert(0->l),
  - layerwise linear-probe accuracy per block (the ground truth for
    "best probing depth").

Deliverable question: does the certified path-supported spectrum
predict the best probing layer / tunnel onset better than the effective
rank baseline?  (CKA and further baselines are added in v2.)

Usage:
  run_probing_depth.py --model resnet50 --weights IMAGENET1K_V2 \
      --dataset cifar10 --root <cifar root> --out results/probing_depth/resnet50.json
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchvision import models as tv_models

from fmca_av.certificate.block_walk import blockwise_pooled_features
from fmca_av.certificate.coordinates import fit_level_coordinates
from fmca_av.certificate.triplet import certificate_report, compose_edge_operators
from fmca_av.data.cifar import CIFARFiles, CIFARViewTransform, CIFARProbeTransform


def build_model(name: str, weights: str):
    factory = getattr(tv_models, name)
    model = factory(weights=None if weights == "none" else weights)
    model.eval()
    return model


@torch.no_grad()
def collect_features(model, images_uint8, transform, views, device, generator_base, batch=64):
    """Per image: one weak view walk + `views` strong view walks per block."""

    weak = CIFARProbeTransform(False, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], size=224)
    outputs_weak, outputs_strong = None, None
    count = images_uint8.shape[0]
    for start in range(0, count, batch):
        chunk = images_uint8[start:start + batch]
        weak_images = torch.stack([weak(img) for img in chunk]).to(device)
        weak_feats = blockwise_pooled_features(model, weak_images)
        strong_feats = []
        for view in range(views):
            strong_images = torch.stack([
                transform(img, torch.Generator().manual_seed(generator_base + (start + i) * 131 + view))
                for i, img in enumerate(chunk)
            ]).to(device)
            strong_feats.append(blockwise_pooled_features(model, strong_images))
        if outputs_weak is None:
            outputs_weak = [[] for _ in weak_feats]
            outputs_strong = [[] for _ in weak_feats]
        for level in range(len(weak_feats)):
            outputs_weak[level].append(weak_feats[level].double().cpu())
            outputs_strong[level].append(
                torch.stack([strong_feats[v][level] for v in range(views)], dim=1).double().cpu()
            )
    return ([torch.cat(parts) for parts in outputs_weak],
            [torch.cat(parts) for parts in outputs_strong])


def project_top(features, dim, basis=None):
    """PCA projection to a manageable operator dimension per block."""

    centered = features - features.mean(dim=0, keepdim=True)
    if basis is None:
        _, _, v = torch.linalg.svd(centered[: min(4096, centered.shape[0])], full_matrices=False)
        basis = v[:dim].transpose(0, 1)
    return centered @ basis, basis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--weights", default="IMAGENET1K_V2")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--root", required=True)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--operator-dim", type=int, default=64)
    parser.add_argument("--stage-level", action="store_true",
                        help="compose at stage boundaries only (4 factors) instead of every block")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    device = torch.device("cuda")
    model = build_model(arguments.model, arguments.weights).to(device)
    data = CIFARFiles(arguments.root, arguments.dataset, train=True)
    labels = torch.from_numpy(data.labels[: arguments.samples].copy())
    images = data.images[: arguments.samples]
    transform = CIFARViewTransform({
        "size": 224, "min_scale": 0.2, "color_jitter_probability": 0.8,
        "color_jitter_strength": 0.5, "grayscale_probability": 0.2,
        "horizontal_flip_probability": 0.5,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    })
    weak_feats, strong_feats = collect_features(
        model, images, transform, arguments.views, device, generator_base=20260820
    )
    levels = len(weak_feats)
    half = arguments.samples // 2
    dim = arguments.operator_dim

    # Stage B on the first half; Stage C on the second half.
    projected_weak, projected_strong, coordinates = [], [], []
    for level in range(levels):
        pw, basis = project_top(weak_feats[level], dim)
        ps = (strong_feats[level] - strong_feats[level].mean(dim=(0, 1), keepdim=True)) @ basis
        coordinate = fit_level_coordinates(
            torch.cat([pw[:half], ps[:half].flatten(0, 1)]), ridge=1e-2
        )
        projected_weak.append(coordinate.encode(pw[half:]))
        projected_strong.append(coordinate.encode(ps[half:]))
        coordinates.append(coordinate)

    if arguments.stage_level:
        # keep stem + the last block of each stage
        from torchvision import models as _tv
        model_ref = build_model(arguments.model, "none")
        sizes = [len(stage) for stage in (model_ref.layer1, model_ref.layer2, model_ref.layer3, model_ref.layer4)]
        keep = [0]
        offset = 1
        for size in sizes:
            offset += size
            keep.append(offset - 1)
        projected_weak = [projected_weak[i] for i in keep]
        projected_strong = [projected_strong[i] for i in keep]
        levels = len(keep)
    edges = []
    for level in range(levels - 1):
        parent = projected_weak[level]
        children = projected_strong[level + 1]
        edges.append(parent.transpose(0, 1) @ children.mean(dim=1) / parent.shape[0])

    curve = []
    for prefix in range(1, levels):
        c_comp = compose_edge_operators(edges[:prefix])
        leaf = projected_strong[prefix]
        c_dir = projected_weak[0].transpose(0, 1) @ leaf.mean(dim=1) / leaf.shape[0]
        report = certificate_report(c_dir, c_comp=c_comp, top_k=8)
        composed_top = float(report.path_singular_values.max())
        # Layerwise probe on the evaluation half's weak features.
        probe_x = projected_weak[prefix - 1] if prefix - 1 < levels else projected_weak[-1]
        curve.append({
            "prefix": prefix,
            "composed_top": composed_top,
            "retention": composed_top / (curve[-1]["composed_top"] + 1e-12) if curve else composed_top,
            "certified_top": float(report.certified_spectrum.max()),
            "certified_sum8": float(report.certified_spectrum[:8].sum()),
            "endpoint_top": float(report.endpoint_singular_values.max()),
            "delta_op": report.delta_operator,
            "normalized_defect": report.delta_frobenius
            / (float(torch.linalg.matrix_norm(report.c_dir)) + 1e-12),
        })

    # Layerwise linear probes (ridge classifier on frozen features).
    from sklearn.linear_model import LogisticRegression

    probe_accs = []
    eval_labels = labels[half:].numpy()
    fit_labels = labels[half:][: half // 2].numpy()
    for level in range(levels):
        x = projected_weak[level].numpy()
        classifier = LogisticRegression(max_iter=500, C=1.0).fit(x[: half // 2], fit_labels)
        probe_accs.append(float(classifier.score(x[half // 2:], eval_labels[half // 2:])))

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": arguments.model,
        "weights": arguments.weights,
        "levels": levels,
        "certificate_curve": curve,
        "layer_probe_accuracy": probe_accs,
        "best_probe_layer": int(np.argmax(probe_accs)),
    }, indent=2))
    print(json.dumps({"best_probe_layer": int(np.argmax(probe_accs)),
                      "probe_accs": [round(a, 3) for a in probe_accs]}, indent=2))


if __name__ == "__main__":
    main()
