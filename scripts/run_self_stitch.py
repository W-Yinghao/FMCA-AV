"""Pilot A1: same-model self-stitch / block bypass.

The cleanest chain in the strategy document: H_i -> H_j -> H_L are all
states of one real network, and H_j is a deterministic successor of
H_i, so the Markov condition holds by construction and no view has to
be invented.

For each interval (i, j) the script reports the path-aware triplet
alongside the local baselines every stitching paper already uses --
linear-regression MSE, CKA, CCA spectrum, endpoint-only score -- and
then performs the ACTUAL intervention: solve a linear bypass on the
calibration split, splice it in place of blocks i+1..j, and measure
what the spliced network does.

The question the pilot exists to answer: is there an interval where the
local metrics look fine and the full-matrix path composition does not,
and does the bypass then actually fail?  A local score that cannot see
the failure is the point of the framework.

Nothing here is a certificate.  These are Gram-corrected point
estimates with a shuffled null, reported as a vector.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchvision import models as tv_models

from fmca_av.certificate.chainspec import ChainSpec, measure_chain, shuffled_null
from fmca_av.data.cifar import CIFARFiles, CIFARProbeTransform


def build_blocks(model_name: str, weights: str):
    """The network as an ordered list of blocks we can slice and splice."""

    model = getattr(tv_models, model_name)(weights=None if weights == "none" else weights)
    model.eval()
    stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    blocks = [stem]
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        blocks.extend(list(stage))
    return model, torch.nn.Sequential(*blocks)


@torch.no_grad()
def block_activations(blocks, images, device, batch=64):
    """Spatially pooled activation after every block, plus the raw last one."""

    pooled = [[] for _ in range(len(blocks))]
    for start in range(0, images.shape[0], batch):
        chunk = images[start:start + batch].to(device)
        value = chunk
        for index, block in enumerate(blocks):
            value = block(value)
            pooled[index].append(value.mean(dim=(2, 3)).double().cpu())
    return [torch.cat(parts) for parts in pooled]


def local_baselines(source: Tensor, target: Tensor, top_k: int = 8) -> dict:
    """The scores a stitching paper would normally stop at."""

    a = (source - source.mean(0, keepdim=True)).double()
    b = (target - target.mean(0, keepdim=True)).double()
    n = a.shape[0]

    # Ridge-regression reconstruction of the target from the source.
    gram = a.transpose(0, 1) @ a / n
    ridge = 1e-3 * float(gram.diagonal().mean())
    weights = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], dtype=torch.float64),
        a.transpose(0, 1) @ b / n,
    )
    residual = b - a @ weights
    mse = float((residual ** 2).sum(dim=1).mean())
    relative = mse / float((b ** 2).sum(dim=1).mean())

    # Linear CKA.
    cross = float((a.transpose(0, 1) @ b).pow(2).sum())
    denom = float(torch.linalg.matrix_norm(a.transpose(0, 1) @ a, ord="fro")
                  * torch.linalg.matrix_norm(b.transpose(0, 1) @ b, ord="fro"))
    cka = cross / denom if denom > 0 else 0.0

    # Canonical correlations.
    def whiten(matrix):
        cov = matrix.transpose(0, 1) @ matrix / n
        cov = cov + 1e-3 * float(cov.diagonal().mean()) * torch.eye(
            cov.shape[0], dtype=torch.float64)
        values, vectors = torch.linalg.eigh(cov)
        return (vectors * values.clamp_min(1e-12).rsqrt()) @ vectors.transpose(0, 1)

    canonical = torch.linalg.svdvals(
        whiten(a) @ (a.transpose(0, 1) @ b / n) @ whiten(b)
    ).clamp(0.0, 1.0)
    return {
        "reconstruction_mse": mse,
        "relative_mse": relative,
        "linear_cka": cka,
        "cca_top": float(canonical[0]),
        "cca_mean_top_k": float(canonical[:top_k].mean()),
    }


def solve_bypass(source: Tensor, target: Tensor, ridge: float = 1e-3):
    """Closed-form linear map replacing the blocks between the two states."""

    a = source.double()
    b = target.double()
    mean_a = a.mean(0, keepdim=True)
    centered = a - mean_a
    gram = centered.transpose(0, 1) @ centered / a.shape[0]
    lam = ridge * float(gram.diagonal().mean())
    weights = torch.linalg.solve(
        gram + lam * torch.eye(gram.shape[0], dtype=torch.float64),
        centered.transpose(0, 1) @ (b - b.mean(0, keepdim=True)) / a.shape[0],
    )
    return weights, mean_a, b.mean(0, keepdim=True)


@torch.no_grad()
def spliced_endpoint(blocks, images, device, low, high, bypass, batch=64):
    """Run the network with blocks low+1..high replaced by the linear map.

    The bypass acts on pooled features, so the spliced path is pooled
    from ``low`` onward.  That is a real intervention on this network's
    computation, and it is also weaker than the blocks it replaces --
    the honest framing is a capacity-matched probe of whether the
    interval is linearly bridgeable, not a claim that the blocks were
    useless.
    """

    weights, mean_source, mean_target = bypass
    outputs = []
    for start in range(0, images.shape[0], batch):
        value = images[start:start + batch].to(device)
        for index, block in enumerate(blocks):
            if low < index <= high:
                continue
            if index == high + 1:
                pooled = value.mean(dim=(2, 3)).double().cpu()
                value = ((pooled - mean_source) @ weights + mean_target)
                value = value.float().to(device)[:, :, None, None]
                value = value.expand(-1, -1, 4, 4)
            value = block(value) if value.ndim == 4 else value
        outputs.append(value.mean(dim=(2, 3)).double().cpu() if value.ndim == 4 else value.double().cpu())
    return torch.cat(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--weights", default="IMAGENET1K_V2")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--root", required=True)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--intervals", default="1:3,1:5,3:8,5:12,8:14",
                        help="comma-separated low:high block index pairs")
    parser.add_argument("--coordinate-budget", type=int, default=64,
                        help="matched per-level coordinate budget; 0 uses native widths")
    parser.add_argument("--variance-floor", type=float, default=1e-2,
                        help="drop directions below floor*lambda_max; bounds the Gram")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, blocks = build_blocks(arguments.model, arguments.weights)
    blocks = blocks.to(device)

    data = CIFARFiles(arguments.root, arguments.dataset, train=True)
    transform = CIFARProbeTransform(False, [0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225], size=224)
    images = torch.stack([transform(img) for img in data.images[: arguments.samples]])
    activations = block_activations(blocks, images, device)
    half = arguments.samples // 2
    endpoint = len(activations) - 1

    records = []
    for pair in arguments.intervals.split(","):
        low, high = (int(value) for value in pair.split(":"))
        if not 0 <= low < high < endpoint:
            print(f"skipping {pair}: needs 0 <= low < high < {endpoint}")
            continue
        spec = ChainSpec(
            name=f"{arguments.model}:{low}->{high}->{endpoint}",
            interpretation="depth_sufficiency",
            state_names=[f"block{low}", f"block{high}", f"block{endpoint}"],
            deterministic_edges=True,
            coordinate_budget=arguments.coordinate_budget or None,
            variance_floor=arguments.variance_floor or None,
            notes="same-model self-stitch: every state is an activation of one network",
        )
        states = [activations[low], activations[high], activations[endpoint]]
        measurement = measure_chain([s[:half] for s in states],
                                    [s[half:] for s in states], spec)
        null = shuffled_null([s[:half] for s in states], [s[half:] for s in states], spec)

        baselines = local_baselines(activations[low][half:], activations[high][half:])
        bypass = solve_bypass(activations[low][:half], activations[high][:half])
        predicted = (activations[low][half:] - bypass[1]) @ bypass[0] + bypass[2]
        actual = activations[high][half:]
        bypass_relative_error = float(((predicted - actual) ** 2).sum(dim=1).mean()
                                      / (actual ** 2).sum(dim=1).mean())
        # Does the bypass preserve what the ENDPOINT needs, not just the
        # interface it was fitted on?
        endpoint_after = local_baselines(predicted, activations[endpoint][half:])
        endpoint_before = local_baselines(actual, activations[endpoint][half:])

        record = {
            "interval": f"{low}:{high}",
            "measurement": measurement,
            "shuffled_null": {"projection": null["projection"]},
            "local_baselines": baselines,
            "bypass": {
                "interface_relative_error": bypass_relative_error,
                "endpoint_cka_after": endpoint_after["linear_cka"],
                "endpoint_cka_before": endpoint_before["linear_cka"],
                "endpoint_cka_drop": endpoint_before["linear_cka"] - endpoint_after["linear_cka"],
            },
        }
        records.append(record)
        p = measurement["projection"]
        print(f"[{low}:{high}] delta_op={p['delta_operator']:.4f} "
              f"path_top={p['path_top']:.3f} endpoint_top={p['endpoint_top']:.3f} "
              f"| cka={baselines['linear_cka']:.4f} relmse={baselines['relative_mse']:.4f} "
              f"| bypass_err={bypass_relative_error:.4f} "
              f"endpoint_cka_drop={record['bypass']['endpoint_cka_drop']:.4f}")

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": arguments.model, "weights": arguments.weights,
        "dataset": arguments.dataset, "samples": arguments.samples,
        "coordinate_budget": arguments.coordinate_budget or None,
        "variance_floor": arguments.variance_floor or None,
        "records": records,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
