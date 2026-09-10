"""Q5, estimation level: what conditional multi-view buys the ESTIMATE.

The learning-level answer (mview beats 2view on the probe) is already
collected.  This measures the thing the learning result is supposed to
follow from, and which has never been measured: with the PARENT sample
held fixed, resampling the child views produces a different estimate
of the cross-moment every time.  The question is how fast that
dispersion shrinks as the number of child views m grows.

Protocol.  One frozen encoder.  Draw the parents and a POOL of P child
views per parent exactly once, and encode them once; the parent
encodings are then literally constant for the rest of the run.  For
each m in the sweep and each of R repeats, choose m of the P pooled
child views, average, and form the Gram-corrected cross-moment against
the fixed parent state.  Dispersion across repeats is the quantity.

Why the pool rather than re-running the loader per repeat: the root
view is itself a random crop, so redrawing the batch moves the PARENT
too, and parent-view noise does not shrink with m -- a first version of
this script did exactly that and produced an almost flat curve for
that reason alone.  Subsets drawn from a common pool overlap, so the
repeats are not fully independent; that is disclosed, and it biases
toward UNDERSTATING how fast dispersion falls, which is the safe
direction for a prediction that says it should fall.

Reported as the operator-norm and Frobenius standard deviation across
resamples, plus the mean estimate norm so the ratio is readable, plus
the fitted slope of log(dispersion) against log(m).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightning as L

from fmca_av.certificate.coordinates import fit_level_coordinates
from fmca_av.certificate.gate_data import GateDataModule
from fmca_av.certificate.gram import build_correction, corrected_endpoint, gram_matrix
from fmca_av.certificate.hierarchy_module import HierarchyCertificateModule

VARIANT_TAGS = {
    "final_2view": "v1_final_2view", "final_mview": "v2_final_mview",
    "additive_2view": "v3_additive_2view", "additive_mview": "v4_additive_mview",
    "amdim_cross": "v5_amdim_cross", "product_only": "v6_product_only",
    "product_endpoint": "v7_product_endpoint",
    "paper_composition": "paper_composition",
}


@torch.no_grad()
def encode_pool(module, data, device, parents_wanted):
    """Encode the parents and their full child-view pool exactly once."""

    parents, children = [], []
    seen = 0
    for batch in data.train_dataloader():
        chain = [t.to(device) for t in batch["chain"]]
        kids = [t.to(device) for t in batch["children"]]
        parents.append(module.encode_level(chain[0], 0).double().cpu())
        children.append(module.encode_level(kids[0], 1).double().cpu())
        seen += parents[-1].shape[0]
        if seen >= parents_wanted:
            break
    return torch.cat(parents)[:parents_wanted], torch.cat(children)[:parents_wanted]


def cross_moment(parent, child_mean, correction=None):
    a = parent - parent.mean(0, keepdim=True)
    b = child_mean - child_mean.mean(0, keepdim=True)
    raw = a.transpose(0, 1) @ b / a.shape[0]
    return raw if correction is None else corrected_endpoint(raw, correction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", default="product_endpoint")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--parents", type=int, default=2000)
    parser.add_argument("--views", default="1,2,4,8",
                        help="child-view counts m to sweep; each must be <= the config's budget")
    parser.add_argument("--resamples", type=int, default=12)
    parser.add_argument("--pool", type=int, default=16,
                        help="child views generated per parent; m is drawn from this pool")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    config = json.loads(
        (Path(arguments.config_dir) / f"gate1_cifar10_{VARIANT_TAGS[arguments.variant]}.json").read_text())
    config["seed"] = arguments.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    config["data"]["view_tree"]["children_per_edge"] = int(arguments.pool)
    sweep = [int(v) for v in arguments.views.split(",") if int(v) <= arguments.pool]
    if not sweep:
        raise SystemExit(f"no requested m fits the pool of {arguments.pool}")

    L.seed_everything(arguments.seed, workers=True)
    data = GateDataModule(config["data"], arguments.seed)
    data.setup()
    parent, pool = encode_pool(module, data, device, arguments.parents)
    print(f"pool encoded once: parents {tuple(parent.shape)}, child views {tuple(pool.shape)}")
    # Parent coordinates are fitted ONCE on the fixed parent sample, so the
    # parent side contributes no variation at all across repeats.
    coords_p = fit_level_coordinates(parent, ridge=1e-3, centered=True)
    zp = coords_p.encode(parent)
    generator = torch.Generator().manual_seed(9_000 + arguments.seed)

    records = []
    for m in sweep:
        estimates = []
        for repeat in range(arguments.resamples):
            picks = torch.stack([torch.randperm(arguments.pool, generator=generator)[:m]
                                 for _ in range(parent.shape[0])])
            child_mean = torch.gather(
                pool, 1, picks.unsqueeze(-1).expand(-1, -1, pool.shape[-1])).mean(dim=1)
            coords_c = fit_level_coordinates(child_mean, ridge=1e-3, centered=True)
            zc = coords_c.encode(child_mean)
            correction = build_correction([gram_matrix(zp), gram_matrix(zc)])
            estimates.append(cross_moment(zp, zc, correction))
        stack = torch.stack(estimates)
        mean = stack.mean(dim=0)
        deviations = stack - mean
        op = [float(torch.linalg.matrix_norm(d, ord=2)) for d in deviations]
        fro = [float(torch.linalg.matrix_norm(d, ord="fro")) for d in deviations]
        record = {
            "views": m, "resamples": arguments.resamples,
            "dispersion_operator": sum(op) / len(op),
            "dispersion_frobenius": sum(fro) / len(fro),
            "mean_operator_norm": float(torch.linalg.matrix_norm(mean, ord=2)),
            "mean_frobenius_norm": float(torch.linalg.matrix_norm(mean, ord="fro")),
        }
        record["relative_dispersion"] = record["dispersion_frobenius"] / record["mean_frobenius_norm"]
        records.append(record)
        print(f"m={m:2d} dispersion_fro={record['dispersion_frobenius']:.5f} "
              f"relative={record['relative_dispersion']:.5f} "
              f"mean_fro={record['mean_frobenius_norm']:.4f}")

    slope = None
    if len(records) >= 2:
        xs = [math.log(r["views"]) for r in records]
        ys = [math.log(max(r["dispersion_frobenius"], 1e-12)) for r in records]
        xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
        denom = sum((x - xm) ** 2 for x in xs)
        slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom if denom else None
        print(f"log-log slope of dispersion vs m: {slope:.3f}  (1/m => -1.0, 1/sqrt(m) => -0.5)")

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "variant": arguments.variant, "seed": arguments.seed,
        "parents": arguments.parents, "resamples": arguments.resamples,
        "child_pool": arguments.pool,
        "backbone_fingerprint": fingerprint, "gram_corrected": True,
        "records": records, "loglog_slope": slope,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
