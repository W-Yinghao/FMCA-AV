"""Evaluation in frozen coordinates, exactly as the supplement's §1.5.

Freeze the encoder and projectors.  On a SEPARATE calibration sample of
roots, estimate level means, Gram matrices, retained eigenspaces and
the truncated transforms.  Hold those fixed while estimating the edge
and endpoint matrices on INDEPENDENT evaluation roots, with separate
edge and endpoint draws.  Evaluation samples never redefine the feature
spaces.

Reported together, as the supplement requires: retained ranks, the
cutoff, the sampling budgets, the endpoint norm, and the closure error.
The relative error is ||Cdir - Ccomp||_F / ||Cdir||_F -- a ratio of
norms, not of squares -- and is UNDEFINED, not zero, when the
denominator vanishes.

This is not the ridge certificate.  A ridge-whitened evaluation of a
truncated-whitened model would report a different operator estimate,
which is the distinction the supplement draws explicitly.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightning as L

from fmca_av.certificate.gate_data import GateDataModule
from fmca_av.certificate.hierarchy_module import HierarchyCertificateModule
from fmca_av.certificate.objective import truncated_whitener
from run_gate1_unit import VARIANT_TAGS


@torch.no_grad()
def collect(module, loader, device, wanted):
    """Raw projected vectors per level, plus endpoint draws, up to `wanted` roots."""

    chain, children, endpoint = None, None, []
    seen = 0
    for batch in loader:
        features = module.feature_batch(
            {k: ([t.to(device) for t in v] if isinstance(v, list) else v.to(device))
             for k, v in batch.items()})
        if chain is None:
            chain = [[] for _ in features.chain]
            children = [[] for _ in features.children]
        for i, value in enumerate(features.chain):
            chain[i].append(value.double().cpu())
        for i, value in enumerate(features.children):
            children[i].append(value.double().cpu())
        endpoint.append(features.endpoint_descendants.double().cpu())
        seen += features.chain[0].shape[0]
        if seen >= wanted:
            break
    cut = lambda parts: torch.cat(parts)[:wanted]
    return [cut(p) for p in chain], [cut(p) for p in children], cut(endpoint)


def level_pool(chain, children, endpoint, level, levels):
    """Supplement Eq. (1): what belongs to level t's marginal statistics."""

    parts = [chain[level]]
    if level >= 1:
        parts.append(children[level - 1].flatten(0, 1))
    if level == levels - 1:
        parts.append(endpoint.flatten(0, 1))
    return torch.cat(parts, dim=0)


def require_accelerator(allow_cpu: bool) -> "torch.device":
    """Pick the device, and refuse a silent CPU fallback.

    These runners encode every sample through the backbone; the matrix
    algebra around it is 128x128 and free.  Landing on the CPU therefore
    costs two orders of magnitude and looks exactly like a slow job rather
    than a misplaced one -- five sweeps reached N=2500 of 20000 in eight
    and a half hours before that was noticed.  Opting into CPU is fine;
    doing it by accident is not.
    """

    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        print("WARNING: running the encoder on CPU because --allow-cpu was given")
        return torch.device("cpu")
    raise SystemExit(
        "no CUDA device: this runner encodes through the backbone and is "
        "orders of magnitude slower on CPU. Submit it with a --gres=gpu:1 "
        "partition (scripts/launch_analysis_gpu.sbatch), or pass --allow-cpu "
        "if you really mean it."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--calibration-roots", type=int, default=10000)
    parser.add_argument("--evaluation-roots", type=int, default=10000)
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="opt in to CPU; the encoder is far slower there")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    config = json.loads(
        (Path(arguments.config_dir) / f"gate1_cifar10_{VARIANT_TAGS[arguments.variant]}.json").read_text())
    config["seed"] = arguments.seed
    device = require_accelerator(arguments.allow_cpu)

    unit = Path(arguments.output_root) / "units" / f"{arguments.variant}__seed{arguments.seed}"
    checkpoint = unit / "checkpoints" / "last.ckpt"
    if not checkpoint.is_file():
        raise SystemExit(f"no checkpoint at {checkpoint}")
    module = HierarchyCertificateModule(config)
    module.load_state_dict(torch.load(str(checkpoint), map_location="cpu",
                                      weights_only=False)["state_dict"])
    module = module.to(device).eval()
    fingerprint = float(sum(float(p.detach().double().abs().sum())
                            for p in module.backbone.parameters()))
    print(f"backbone fingerprint {fingerprint:.6f}")

    levels = module.num_levels
    # Stage B: calibration roots fix means, Grams, retained spaces, transforms.
    L.seed_everything(arguments.seed, workers=True)
    data = GateDataModule(config["data"], arguments.seed)
    data.setup()
    chain, children, endpoint = collect(module, data.train_dataloader(), device,
                                        arguments.calibration_roots)
    means, whiteners, ranks = [], [], []
    for level in range(levels):
        pooled = level_pool(chain, children, endpoint, level, levels)
        mean = pooled.mean(dim=0, keepdim=True)
        centered = pooled - mean
        gram = centered.transpose(0, 1) @ centered / centered.shape[0]
        whitener, retained = truncated_whitener(gram, arguments.tau)
        if whitener is None:
            raise SystemExit(f"level {level}: empty retained set on the calibration sample")
        means.append(mean); whiteners.append(whitener); ranks.append(retained)
    print(f"Stage B: retained ranks {ranks} at tau={arguments.tau}")

    # Stage C: independent evaluation roots, frozen coordinates.
    L.seed_everything(arguments.seed + 991, workers=True)
    data_eval = GateDataModule(config["data"], arguments.seed + 991)
    data_eval.setup()
    e_chain, e_children, e_endpoint = collect(module, data_eval.train_dataloader(), device,
                                              arguments.evaluation_roots)
    z = [(e_chain[i] - means[i]) @ whiteners[i] for i in range(levels)]
    zc = [(e_children[i] - means[i + 1].unsqueeze(0)) @ whiteners[i + 1] for i in range(levels - 1)]
    ze = (e_endpoint - means[-1].unsqueeze(0)) @ whiteners[-1]

    n = z[0].shape[0]
    edges = [z[i].transpose(0, 1) @ zc[i].mean(dim=1) / n for i in range(levels - 1)]
    c_comp = edges[0]
    for edge in edges[1:]:
        c_comp = c_comp @ edge
    c_dir = z[0].transpose(0, 1) @ ze.mean(dim=1) / n

    endpoint_norm = float(torch.linalg.matrix_norm(c_dir, ord="fro"))
    residual = float(torch.linalg.matrix_norm(c_dir - c_comp, ord="fro"))
    relative = residual / endpoint_norm if endpoint_norm > 0 else None
    singular = torch.linalg.svdvals(c_dir)

    record = {
        "variant": arguments.variant, "seed": arguments.seed,
        "backbone_fingerprint": fingerprint, "estimator": "truncated",
        "tau": arguments.tau, "retained_ranks": ranks,
        "calibration_roots": arguments.calibration_roots,
        "evaluation_roots": arguments.evaluation_roots,
        "children_per_edge": int(config["data"]["view_tree"]["children_per_edge"]),
        "endpoint_descendants": int(config["data"]["view_tree"]["endpoint_descendants"]),
        "endpoint_norm_frobenius": endpoint_norm,
        "closure_residual_frobenius": residual,
        "relative_closure_error": relative,
        "composed_norm_frobenius": float(torch.linalg.matrix_norm(c_comp, ord="fro")),
        "endpoint_operator_norm": float(singular.max()),
        "endpoint_singular_values": [float(v) for v in singular[:16]],
        "edge_operator_norms": [float(torch.linalg.matrix_norm(e, ord=2)) for e in edges],
    }
    print(f"endpoint ||Cdir||_F={endpoint_norm:.4f}  residual={residual:.4f}  "
          f"relative={'undefined' if relative is None else f'{relative:.4f}'}  "
          f"top sigma={float(singular.max()):.4f}")
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
