"""Pilot A3: deterministic pretrained depth measurement.

The strategy's §7.3 and §5.1: measure the network's own activation
chain H_0 -> ... -> H_L, where every edge is the deterministic function
the network already computes.  No view has to be resampled and no
crop severity has to be aligned to a tap, so the estimand is the
network's depth structure rather than a channel we chose for it.

What the finite projected defect means here: whether the coordinates
retained at each stage are sufficient for the suffix.  The calibration
distribution is part of the estimand, so the same model is measured on
two domains and the two profiles are reported side by side rather than
averaged.

Reported as a vector, with a pairing-shuffled null.  No certificate
language.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchvision import models as tv_models

from fmca_av.certificate.chainspec import ChainSpec, measure_chain, shuffled_null
from fmca_av.data.cifar import CIFARFiles, CIFARProbeTransform


@torch.no_grad()
def stage_activations(model, images, device, batch=64):
    """Pooled activation after the stem and each of the four stages."""

    stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    collected = [[] for _ in range(len(stages) + 1)]
    for start in range(0, images.shape[0], batch):
        value = stem(images[start:start + batch].to(device))
        collected[0].append(value.mean(dim=(2, 3)).double().cpu())
        for index, stage in enumerate(stages):
            value = stage(value)
            collected[index + 1].append(value.mean(dim=(2, 3)).double().cpu())
    return [torch.cat(parts) for parts in collected]


def probe_accuracy(train_x, train_y, test_x, test_y, classes, device, iterations=150):
    """Convex probe, so the number is a property of the features."""

    mean, std = train_x.mean(0, keepdim=True), train_x.std(0, keepdim=True).clamp_min(1e-6)
    xtr = ((train_x - mean) / std).float().to(device)
    xte = ((test_x - mean) / std).float().to(device)
    ytr, yte = train_y.to(device), test_y.to(device)
    weight = torch.zeros(xtr.shape[1], classes, device=device, requires_grad=True)
    bias = torch.zeros(classes, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=iterations,
                                  history_size=10, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(xtr @ weight + bias, ytr) \
            + 1e-4 * (weight * weight).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return float(((xte @ weight + bias).argmax(dim=1) == yte).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--weights", default="IMAGENET1K_V2")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--root", required=True)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--coordinate-budget", type=int, default=64,
                        help="matched per-level coordinate budget; 0 uses native widths")
    parser.add_argument("--variance-floor", type=float, default=1e-2,
                        help="drop directions below floor*lambda_max; bounds the Gram")
    parser.add_argument("--seed", type=int, default=1,
                        help="draw index: seed s takes samples [(s-1)*N, s*N), so "
                             "seeds are DISJOINT draws and seed 1 is the first block")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(tv_models, arguments.model)(
        weights=None if arguments.weights == "none" else arguments.weights).eval().to(device)

    data = CIFARFiles(arguments.root, arguments.dataset, train=True)
    transform = CIFARProbeTransform(False, [0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225], size=224)
    lo = (arguments.seed - 1) * arguments.samples
    hi = lo + arguments.samples
    if hi > len(data.images):
        raise SystemExit(f"seed {arguments.seed} needs images [{lo}:{hi}] of {len(data.images)}")
    images = torch.stack([transform(img) for img in data.images[lo:hi]])
    labels = torch.from_numpy(data.labels[lo:hi].copy())
    states = stage_activations(model, images, device)
    half = arguments.samples // 2
    classes = 100 if arguments.dataset == "cifar100" else 10

    names = ["stem", "layer1", "layer2", "layer3", "layer4"]
    chains = []
    # Every contiguous sub-chain that ends at the endpoint: "is the
    # suffix reachable from here through the interfaces in between".
    for start in range(len(states) - 2):
        subset = list(range(start, len(states)))
        spec = ChainSpec(
            name=f"{arguments.model}/{arguments.dataset}:{names[start]}->endpoint",
            interpretation="depth_sufficiency",
            state_names=[names[i] for i in subset],
            deterministic_edges=True,
            coordinate_budget=arguments.coordinate_budget or None,
            variance_floor=arguments.variance_floor or None,
            notes="the network's own activation chain; no view is resampled",
        )
        calibration = [states[i][:half] for i in subset]
        evaluation = [states[i][half:] for i in subset]
        try:
            measurement = measure_chain(calibration, evaluation, spec)
            null = shuffled_null(calibration, evaluation, spec)
        except ValueError as refusal:
            # A refusal IS a result for a control arm: a random-init pooled
            # chain can be near rank-1, and the coordinate rule must say so
            # in the record rather than dying with an empty output file.
            chains.append({"start": names[start],
                           "states": [names[i] for i in subset],
                           "refused": str(refusal)})
            print(f"{names[start]:8s}->endpoint  REFUSED: {refusal}")
            continue
        p = measurement["projection"]
        chains.append({
            "start": names[start],
            "states": [names[i] for i in subset],
            "projection": p,
            "surrogate": measurement["surrogate"],
            "null_endpoint_top": null["projection"]["endpoint_top"],
            "null_delta_operator": null["projection"]["delta_operator"],
            "gram": measurement["gram"],
            "dimensions": measurement["dimensions"],
            "native_dimensions": measurement["native_dimensions"],
            "budget_retained_variance": measurement["budget_retained_variance"],
        })
        print(f"{names[start]:8s}->endpoint  delta_op={p['delta_operator']:.4f} "
              f"path_top={p['path_top']:.3f} endpoint_top={p['endpoint_top']:.3f} "
              f"eff_rank={p['endpoint_effective_rank']:.1f} "
              f"| null endpoint_top={null['projection']['endpoint_top']:.3f}")

    quarter = half // 2
    probes = []
    for index, name in enumerate(names):
        accuracy = probe_accuracy(states[index][half:half + quarter], labels[half:half + quarter],
                                  states[index][half + quarter:], labels[half + quarter:],
                                  classes, device)
        probes.append({"stage": name, "probe_accuracy": accuracy})
        print(f"  probe {name:8s} {accuracy * 100:.2f}%")

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": arguments.model, "weights": arguments.weights,
        "dataset": arguments.dataset, "samples": arguments.samples,
        "seed": arguments.seed, "draw": [lo, hi],
        "coordinate_budget": arguments.coordinate_budget or None,
        "variance_floor": arguments.variance_floor or None,
        "chains": chains, "layer_probes": probes,
        "best_probe_stage": max(probes, key=lambda r: r["probe_accuracy"])["stage"],
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
