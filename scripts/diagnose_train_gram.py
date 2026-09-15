"""Why the Gram-corrected closure diverges: measure, do not guess.

The type-2 probe cleared the device fault and then tripped the
divergence guard.  The suspected cause is the one the chain track
already met: with batch-level Grams the inverse in the corrected
composition amplifies directions the ridge has already destroyed.
This reports the batch Gram spectrum of an EXISTING trained V7
checkpoint -- no training, no new arm -- so the conditioning rule for
the training path is chosen from a measurement and can be frozen
before any fleet runs.
"""

import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lightning as L
from fmca_av.certificate.gate_data import GateDataModule
from fmca_av.certificate.gram import (
    build_correction, corrected_composition, corrected_endpoint,
    gram_matrix, spectral_pseudo_inverse)
from fmca_av.certificate.hierarchy_module import HierarchyCertificateModule
from fmca_av.certificate.objective import (
    compose_edge_operators, train_edge_operators, train_endpoint_operator, whiten_chain_batch)
from run_gate1_unit import VARIANT_TAGS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--variant", default="product_endpoint")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--taus", default="1e-3,1e-2,5e-2,1e-1,2e-1")
    p.add_argument("--random-init", action="store_true",
                   help="skip the checkpoint: the probe dies in the FIRST epoch, "
                        "so the state that matters is the untrained one")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    config = json.loads((Path(a.config_dir) / f"gate1_cifar10_{VARIANT_TAGS[a.variant]}.json").read_text())
    config["seed"] = a.seed
    L.seed_everything(a.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = HierarchyCertificateModule(config)
    if not a.random_init:
        ckpt = Path(a.output_root) / "units" / f"{a.variant}__seed{a.seed}" / "checkpoints" / "last.ckpt"
        if not ckpt.is_file():
            raise SystemExit(f"no checkpoint at {ckpt}")
        module.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=False)["state_dict"])
    print(f"weights_loaded={not a.random_init}")
    module = module.to(device).eval()
    data = GateDataModule(config["data"], a.seed); data.setup()

    taus = [float(v) for v in a.taus.split(",")]
    rows = []
    with torch.no_grad():
        for index, batch in enumerate(data.train_dataloader()):
            if index >= a.batches: break
            batch = {k: ([t.to(device) for t in v] if isinstance(v, list) else v.to(device))
                     for k, v in batch.items()}
            features = module.feature_batch(batch)
            whitened, _, _ = whiten_chain_batch(features, ridge=module.ridge,
                                             detach_whitener=module.detach_whitener)
            states = list(whitened.chain[: module.num_levels - 1])
            states.append(whitened.endpoint_descendants.mean(dim=1))
            grams = [gram_matrix(s.double()) for s in states]
            edges = train_edge_operators(whitened)
            c_comp = compose_edge_operators(edges)
            c_dir = train_endpoint_operator(whitened)
            surrogate = float(torch.linalg.matrix_norm(c_comp, ord="fro"))
            # The two closure ratios the loss would actually see, side by
            # side: this is what says whether the corrected term enters at a
            # different WEIGHT, which would confound geometry with weighting.
            correction = build_correction([g for g in grams])
            gc = corrected_composition([e.double() for e in edges], correction)
            gd = corrected_endpoint(c_dir.double(), correction)
            surrogate_closure = float((c_dir - c_comp).square().sum() / c_dir.square().sum())
            corrected_closure = float((gd - gc).square().sum() / gd.square().sum())
            row = {"batch": index, "surrogate_comp_fro": surrogate,
                   "corrected_comp_fro": float(torch.linalg.matrix_norm(gc, ord="fro")),
                   "corrected_dir_fro": float(torch.linalg.matrix_norm(gd, ord="fro")),
                   "surrogate_closure_ratio": surrogate_closure,
                   "corrected_closure_ratio": corrected_closure,
                   "closure_inflation": corrected_closure / max(surrogate_closure, 1e-12),
                   "levels": []}
            for level, g in enumerate(grams):
                values = torch.linalg.eigvalsh(g).clamp_min(0.0)
                identity = torch.eye(g.shape[0], dtype=torch.float64, device=g.device)
                row["levels"].append({
                    "level": level,
                    "deviation": float(torch.linalg.matrix_norm(g - identity, ord=2)),
                    "min_eig": float(values.min()), "max_eig": float(values.max()),
                    "retained": {f"{t:g}": spectral_pseudo_inverse(g, -1.0, t)[1] for t in taus},
                    "amplification": {f"{t:g}": float(torch.linalg.matrix_norm(
                        spectral_pseudo_inverse(g, -1.0, t)[0], ord=2)) for t in taus},
                    "dimension": int(g.shape[0]),
                })
            rows.append(row)
            print(f"batch {index}: closure surrogate={surrogate_closure:.4f} "
                  f"corrected={corrected_closure:.4f} inflation=x{row['closure_inflation']:.2f} "
                  f"| ||C_comp||_F {surrogate:.2f}->{row['corrected_comp_fro']:.2f} "
                  f"||C_dir||_F ->{row['corrected_dir_fro']:.2f}")
            for lv in row["levels"]:
                amp = "  ".join(f"tau={k}: x{v:.1f}({lv['retained'][k]}/{lv['dimension']})"
                                for k, v in lv["amplification"].items())
                print(f"  level {lv['level']}: dev={lv['deviation']:.4f} "
                      f"eig[{lv['min_eig']:.2e},{lv['max_eig']:.3f}]  {amp}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"variant": a.variant, "seed": a.seed, "taus": taus, "batches": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
