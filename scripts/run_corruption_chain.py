"""Corruption-path pilot: the first real data for that interpretation.

ImageNet-C serves the SAME validation image at increasing severities of
one named corruption, so the chain

    clean -> severity s_1 -> ... -> severity s_k

is a real, physically ordered degradation process with exact per-image
pairing.  Every state is the frozen encoder's feature of the image at
that severity; the finite projected defect asks whether the dependence
between the clean image and the worst corruption is mediated by the
intermediate severities -- approximate lumpability of the corruption
process in feature space.

The severity ladder is parameterized, not literally recursive
(severity 3 is not the corruption applied to severity 1), so
``deterministic_edges=False`` and the interpretation is
``corruption_path``: a statement about the process's feature-space
Markov structure, not about any physical channel.

The clean state uses the standard eval transform; ImageNet-C ships
pre-sized at 224.  That asymmetry belongs to the benchmark and is
recorded here rather than hidden.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchvision import models as tv_models

from fmca_av.certificate.chainspec import ChainSpec, measure_chain, shuffled_null
from fmca_av.data.imagenet_files import ImageNetCFiles, ImageNetValFiles


@torch.no_grad()
def encode(model, images, device, batch=64):
    """Penultimate (pooled layer4) features of a frozen classifier."""

    parts = []
    for start in range(0, images.shape[0], batch):
        chunk = images[start:start + batch].to(device)
        value = model.maxpool(model.relu(model.bn1(model.conv1(chunk))))
        for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
            value = stage(value)
        parts.append(value.mean(dim=(2, 3)).double().cpu())
    return torch.cat(parts)


def probe_accuracy(train_x, train_y, test_x, test_y, classes, device, iterations=150):
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
    parser.add_argument("--corruption", required=True)
    parser.add_argument("--severities", default="1,3,5")
    parser.add_argument("--imagenet-root",
                        default="/projects/EEG-foundation-model/yinghao/FMCA-AV/imagenet")
    parser.add_argument("--imagenet-c-root",
                        default="/projects/EEG-foundation-model/yinghao/FMCA-AV/robustness/imagenet-c")
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--coordinate-budget", type=int, default=64)
    parser.add_argument("--gram-bound", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1,
                        help="draw index: seed s takes images [(s-1)*N, s*N)")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(tv_models, arguments.model)(weights=arguments.weights).eval().to(device)

    clean = ImageNetValFiles(arguments.imagenet_root)
    lo = (arguments.seed - 1) * arguments.samples
    hi = lo + arguments.samples
    if hi > len(clean):
        raise SystemExit(f"seed {arguments.seed} needs images [{lo}:{hi}] of {len(clean)}")

    severities = [int(v) for v in arguments.severities.split(",")]
    names = ["clean"] + [f"sev{v}" for v in severities]

    # One severity in memory at a time: load, encode, free.
    images, labels = clean.load_block(lo, hi)
    states = [encode(model, images, device)]
    del images
    for severity in severities:
        corrupted = ImageNetCFiles(arguments.imagenet_c_root, arguments.corruption,
                                   severity, clean).load_block(lo, hi)
        states.append(encode(model, corrupted, device))
        del corrupted

    half = arguments.samples // 2
    spec = ChainSpec(
        name=f"{arguments.model}/{arguments.corruption}:{'->'.join(names)}",
        interpretation="corruption_path",
        state_names=names,
        deterministic_edges=False,
        coordinate_budget=arguments.coordinate_budget or None,
        gram_bound=arguments.gram_bound or None,
        notes=("ImageNet-C severity ladder, exact per-image pairing; parameterized "
               "severities, not recursive application; clean uses the standard eval "
               "transform, ImageNet-C ships pre-sized at 224"),
    )
    calibration = [s[:half] for s in states]
    evaluation = [s[half:] for s in states]
    measurement = measure_chain(calibration, evaluation, spec, with_radii=True)
    null = shuffled_null(calibration, evaluation, spec)
    p = measurement["projection"]
    print(f"{arguments.corruption} {'->'.join(names)}: delta_op={p['delta_operator']:.4f} "
          f"endpoint_top={p['endpoint_top']:.3f} path_top={p['path_top']:.3f} "
          f"| null delta_op={null['projection']['delta_operator']:.4f}")

    quarter = half // 2
    probes = []
    for index, name in enumerate(names):
        accuracy = probe_accuracy(states[index][half:half + quarter], labels[half:half + quarter],
                                  states[index][half + quarter:], labels[half + quarter:],
                                  clean.classes, device)
        probes.append({"state": name, "probe_accuracy": accuracy})
        print(f"  probe {name:8s} {accuracy * 100:.2f}%")

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": arguments.model, "weights": arguments.weights,
        "corruption": arguments.corruption, "severities": severities,
        "samples": arguments.samples, "seed": arguments.seed, "draw": [lo, hi],
        "coordinate_budget": arguments.coordinate_budget or None,
        "gram_bound": arguments.gram_bound or None,
        "measurement": measurement,
        "shuffled_null": {"projection": null["projection"]},
        "state_probes": probes,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
