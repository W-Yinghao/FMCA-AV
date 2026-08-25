"""E-D2: the price of chain length at fixed endpoints.

Fix the two ends of the chain -- the stem state and the final block
state -- and vary ONLY how many intermediate taps the path is required
to thread through.  Block-level activations are extracted once; each
chain length reuses them, so the whole scan is one job and every length
sees the identical draw.

What the curve means: C_dir is the same operator at every length, so
the scan isolates how the composed path degrades (or does not) as the
factorization gets finer.  The telescoping notes predict the defect can
grow with interface count when each tap truncates; a flat curve would
say the retained coordinates are closed under the blocks.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmca_av.certificate.chainspec import ChainSpec, measure_chain, shuffled_null
from fmca_av.data.cifar import CIFARFiles, CIFARProbeTransform
from run_self_stitch import build_blocks, block_activations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--weights", default="IMAGENET1K_V2")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--root", required=True)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--lengths", default="1,2,4,8,14",
                        help="numbers of INTERMEDIATE taps between the fixed ends")
    parser.add_argument("--coordinate-budget", type=int, default=64)
    parser.add_argument("--variance-floor", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, blocks = build_blocks(arguments.model, arguments.weights)
    blocks = blocks.to(device).eval()

    data = CIFARFiles(arguments.root, arguments.dataset, train=True)
    transform = CIFARProbeTransform(False, [0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225], size=224)
    lo = (arguments.seed - 1) * arguments.samples
    hi = lo + arguments.samples
    if hi > len(data.images):
        raise SystemExit(f"seed {arguments.seed} needs images [{lo}:{hi}] of {len(data.images)}")
    images = torch.stack([transform(img) for img in data.images[lo:hi]])
    activations = block_activations(blocks, images, device)
    half = arguments.samples // 2
    final = len(activations) - 1
    interior = list(range(1, final))  # candidate intermediate taps

    records = []
    for length in (int(v) for v in arguments.lengths.split(",")):
        if length > len(interior):
            print(f"skipping length {length}: only {len(interior)} interior blocks")
            continue
        # Evenly spaced intermediate taps between the fixed ends.
        picks = [interior[round(i * (len(interior) - 1) / max(length - 1, 1))]
                 for i in range(length)] if length > 1 else [interior[len(interior) // 2]]
        picks = sorted(dict.fromkeys(picks))
        subset = [0] + picks + [final]
        spec = ChainSpec(
            name=f"{arguments.model}/{arguments.dataset}:L={len(subset) - 1}",
            interpretation="depth_sufficiency",
            state_names=[f"block{i}" for i in subset],
            deterministic_edges=True,
            coordinate_budget=arguments.coordinate_budget or None,
            variance_floor=arguments.variance_floor or None,
            notes="fixed ends, varying interface count; one shared draw for every length",
        )
        states = [activations[i] for i in subset]
        measurement = measure_chain([s[:half] for s in states], [s[half:] for s in states],
                                    spec, with_radii=True)
        null = shuffled_null([s[:half] for s in states], [s[half:] for s in states], spec)
        p = measurement["projection"]
        records.append({
            "edges": len(subset) - 1, "taps": subset,
            "projection": p, "surrogate": measurement["surrogate"],
            "gram": measurement["gram"],
            "finite_sample": measurement["finite_sample"],
            "null_delta_operator": null["projection"]["delta_operator"],
            "null_endpoint_top": null["projection"]["endpoint_top"],
        })
        print(f"L={len(subset) - 1:2d} taps={subset} delta_op={p['delta_operator']:.4f} "
              f"path_top={p['path_top']:.3f} endpoint_top={p['endpoint_top']:.3f} "
              f"| null {null['projection']['delta_operator']:.3f} "
              f"r_P={measurement['finite_sample']['path_radius']:.3f}")

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": arguments.model, "weights": arguments.weights,
        "dataset": arguments.dataset, "samples": arguments.samples,
        "seed": arguments.seed, "draw": [lo, hi],
        "coordinate_budget": arguments.coordinate_budget or None,
        "variance_floor": arguments.variance_floor or None,
        "records": records,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
