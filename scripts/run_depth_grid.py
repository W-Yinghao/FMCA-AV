"""The (start depth, end depth) grid: depth resolution and endpoint ablation.

The stage-level profile has three points per cell and every chain ends
at layer4, so "monotone decline toward the endpoint" rests on three
measurements and confounds distance-from-end with absolute position.
This grid removes both limits in one job: block-level activations are
extracted once, and every chain [block b] -> stage boundaries between
-> [boundary e] with at least three states is measured on the same
draw.  Rows of the grid are the fine-resolution profile; columns are
the endpoint ablation.

Intermediate taps stay at stage boundaries -- the L-scan showed that
factoring finer than the calibration budget buys estimator noise, so
the START varies finely while the downstream factorization is held at
stage level.

Per-block convex probes give the fine probe curve alongside.
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
from run_depth_chain import probe_accuracy
from run_self_stitch import block_activations, build_blocks


@torch.no_grad()
def vit_block_states(model, images, device, batch=64):
    """Token-mean state after the embedded input and every encoder block."""

    collected = [[] for _ in range(len(model.encoder.layers) + 1)]
    for start in range(0, images.shape[0], batch):
        chunk = images[start:start + batch].to(device)
        tokens = model._process_input(chunk)
        cls = model.class_token.expand(tokens.shape[0], -1, -1)
        value = torch.cat([cls, tokens], dim=1) + model.encoder.pos_embedding
        value = model.encoder.dropout(value)
        collected[0].append(value[:, 1:].mean(dim=1).double().cpu())
        for index, block in enumerate(model.encoder.layers):
            value = block(value)
            collected[index + 1].append(value[:, 1:].mean(dim=1).double().cpu())
    return [torch.cat(parts) for parts in collected]


def resnet_layout(model):
    lengths = [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)]
    boundaries, total = [], 0
    for length in lengths:
        total += length
        boundaries.append(total)
    return boundaries  # activation indices of stage outputs; 0 is the stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--weights", default="IMAGENET1K_V2")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--root", required=True)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--coordinate-budget", type=int, default=64)
    parser.add_argument("--gram-bound", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(tv_models, arguments.model)(
        weights=None if arguments.weights == "none" else arguments.weights).eval().to(device)

    lo = (arguments.seed - 1) * arguments.samples
    hi = lo + arguments.samples
    if arguments.dataset == "imagenet":
        from fmca_av.data.imagenet_files import ImageNetValFiles
        files = ImageNetValFiles(arguments.root)
        if hi > len(files):
            raise SystemExit(f"seed {arguments.seed} needs images [{lo}:{hi}] of {len(files)}")
        images, labels = files.load_block(lo, hi)
        classes = files.classes
    else:
        data = CIFARFiles(arguments.root, arguments.dataset, train=True)
        transform = CIFARProbeTransform(False, [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225], size=224)
        if hi > len(data.images):
            raise SystemExit(f"seed {arguments.seed} needs images [{lo}:{hi}] of {len(data.images)}")
        images = torch.stack([transform(img) for img in data.images[lo:hi]])
        labels = torch.from_numpy(data.labels[lo:hi].copy())
        classes = 100 if arguments.dataset == "cifar100" else 10

    if arguments.model.startswith("vit"):
        states = vit_block_states(model, images, device)
        boundaries = [3, 6, 9, 12]  # the standard 2/5/8/11 taps as state indices
        state_names = ["embed"] + [f"block{i}" for i in range(len(states) - 1)]
    elif hasattr(model, "layer1"):
        _, blocks = build_blocks(arguments.model, arguments.weights)
        blocks = blocks.to(device).eval()
        states = block_activations(blocks, images, device)
        boundaries = resnet_layout(model)
        state_names = ["stem"] + [f"block{i}" for i in range(1, len(states))]
    else:
        raise SystemExit(f"no block layout for {arguments.model}")

    del images
    half = arguments.samples // 2
    records = []
    for end in boundaries:
        for start in range(end):
            middle = [b for b in boundaries if start < b < end]
            subset = [start] + middle + [end]
            if len(subset) < 3:
                continue
            spec = ChainSpec(
                name=f"{arguments.model}/{arguments.dataset}:{start}->{end}",
                interpretation="depth_sufficiency",
                state_names=[state_names[i] for i in subset],
                deterministic_edges=True,
                coordinate_budget=arguments.coordinate_budget or None,
                gram_bound=arguments.gram_bound or None,
                notes="(start, end) grid; intermediate taps held at stage boundaries",
            )
            chain_states = [states[i] for i in subset]
            try:
                measurement = measure_chain([s[:half] for s in chain_states],
                                            [s[half:] for s in chain_states], spec)
                null = shuffled_null([s[:half] for s in chain_states],
                                     [s[half:] for s in chain_states], spec)
            except ValueError as refusal:
                records.append({"start": start, "end": end, "refused": str(refusal)})
                continue
            records.append({
                "start": start, "end": end, "taps": subset,
                "projection": measurement["projection"],
                "null_delta_operator": null["projection"]["delta_operator"],
                "gram": measurement["gram"],
            })
    for row in records:
        if "refused" not in row:
            print(f"start={row['start']:2d} end={row['end']:2d} "
                  f"d_op={row['projection']['delta_operator']:.4f} "
                  f"null={row['null_delta_operator']:.3f}")

    quarter = half // 2
    probes = []
    for index in range(len(states)):
        accuracy = probe_accuracy(states[index][half:half + quarter], labels[half:half + quarter],
                                  states[index][half + quarter:], labels[half + quarter:],
                                  classes, device)
        probes.append({"state": state_names[index], "index": index, "probe_accuracy": accuracy})

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": arguments.model, "weights": arguments.weights,
        "dataset": arguments.dataset, "samples": arguments.samples,
        "seed": arguments.seed, "draw": [lo, hi],
        "coordinate_budget": arguments.coordinate_budget or None,
        "gram_bound": arguments.gram_bound or None,
        "boundaries": boundaries, "state_names": state_names,
        "grid": records, "block_probes": probes,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
