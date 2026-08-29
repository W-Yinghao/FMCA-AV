"""T0 validity gate on CPU: degenerate controls (E-A1) and the
calibration convergence sweep (E-A2).

E-A1 asks whether a low normalized defect can be bought outright by a
weak encoder.  Two degenerate rows go through the identical Stage-B/C
protocol: a randomly initialized encoder (never trained) and a
deliberately collapsed one (its stage outputs scaled toward a constant).
If either reaches the V7 range, the ratio form is dead and the report
must switch to (numerator, denominator) or a spectrum-resolved form.

E-A2 asks where the measurement stops moving.  The encoder runs once
over the whole held-out pool and Stage-B is truncated afterwards, so
every N sees the same images.  Note the ceiling: n_calibration + n_val
is all that is held out from SSL training, so the clean sweep tops out
there -- anything larger would use images the encoder was trained on.

Both are evaluation only and run fine on CPU.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightning as L

from fmca_av.certificate.gate_data import GateDataModule
from fmca_av.certificate.gram import (
    build_correction,
    corrected_composition,
    corrected_endpoint,
    cumulative_interface_attribution,
    gram_matrix,
)
from fmca_av.certificate.hierarchy_module import HierarchyCertificateModule
from run_gate1_unit import VARIANT_TAGS, certificate_evaluation


class CollapsedBackbone(torch.nn.Module):
    """Wraps a backbone so every stage output shrinks toward its own mean.

    A collapsed encoder is the adversarial case for a ratio: tiny
    operators everywhere, so both terms fall and the quotient can look
    excellent for no representational reason.
    """

    def __init__(self, backbone, scale: float = 0.01) -> None:
        super().__init__()
        self.backbone = backbone
        self.scale = float(scale)
        self.stage_dims = backbone.stage_dims
        self.output_dim = backbone.output_dim

    def forward_stages(self, inputs, up_to=None):
        stages = self.backbone.forward_stages(inputs, up_to=up_to)
        return [value.mean(dim=0, keepdim=True) + self.scale * (value - value.mean(dim=0, keepdim=True))
                for value in stages]

    def forward(self, inputs):
        return self.forward_stages(inputs)[-1]


def summarize(evaluation):
    point = evaluation["point"]
    record = {
        "normalized_closure_defect": evaluation["normalized_closure_defect"],
        "numerator_delta_frobenius": point["delta_frobenius"],
        "denominator_dir_frobenius": evaluation["dir_frobenius"],
        "endpoint_top": max(point["endpoint_singular_values"]),
        "path_top": max(point["path_singular_values"]),
    }
    if "corrected" in evaluation:
        record["corrected"] = evaluation["corrected"]
    return record


def gram_corrected_report(module, data_module, device, seed, measurement_ridge,
                          calibration_limit=0, pool_val=False):
    """Corrected and surrogate defects side by side, plus the artifact split.

    The Gram is a Stage-B object: it is estimated on the same held-out
    pool that fits the coordinates, never on the Stage-C evaluation
    sample.
    """

    from run_gate1_unit import (
        collect_chain_features,
        encode_chain_batch,
        estimate_edge_operators,
        estimate_endpoint_operator,
        level_calibration_features,
    )
    from fmca_av.certificate.coordinates import fit_level_coordinates
    from fmca_av.certificate.estimation import ChainFeatureBatch

    calibration = collect_chain_features(
        module, data_module.calibration_dataloader(), device, max_parents=0
    )
    if pool_val:
        extra = collect_chain_features(
            module, data_module.val_dataloader(), device, max_parents=0
        )
        calibration = ChainFeatureBatch(
            chain=[torch.cat([a, b]) for a, b in zip(calibration.chain, extra.chain)],
            children=[torch.cat([a, b]) for a, b in zip(calibration.children, extra.children)],
            endpoint_descendants=torch.cat(
                [calibration.endpoint_descendants, extra.endpoint_descendants]
            ),
        )
    if calibration_limit:
        keep = int(calibration_limit)
        calibration = ChainFeatureBatch(
            chain=[value[:keep] for value in calibration.chain],
            children=[value[:keep] for value in calibration.children],
            endpoint_descendants=calibration.endpoint_descendants[:keep],
        )
    coordinates = [
        fit_level_coordinates(level_calibration_features(calibration, level),
                              ridge=measurement_ridge)
        for level in range(calibration.num_levels)
    ]
    # Stage-B Gram: the coordinates applied to the same held-out pool.
    grams = [gram_matrix(coordinates[level].encode(
                 level_calibration_features(calibration, level)))
             for level in range(calibration.num_levels)]
    correction = build_correction(grams)

    raw = collect_chain_features(module, data_module.test_dataloader(), device, max_parents=0)
    encoded = encode_chain_batch(raw, coordinates)
    edges = estimate_edge_operators(encoded)
    c_dir = estimate_endpoint_operator(encoded)

    surrogate_comp = edges[0].double()
    for edge in edges[1:]:
        surrogate_comp = surrogate_comp @ edge.double()
    surrogate_delta = c_dir.double() - surrogate_comp
    corrected_comp = corrected_composition(edges, correction)
    corrected_dir = corrected_endpoint(c_dir, correction)
    corrected_delta = corrected_dir - corrected_comp

    def norms(delta, direct):
        numerator = float(torch.linalg.matrix_norm(delta, ord="fro"))
        denominator = float(torch.linalg.matrix_norm(direct, ord="fro"))
        return {"numerator": numerator, "denominator": denominator,
                "ratio": numerator / max(denominator, 1e-12),
                "delta_operator": float(torch.linalg.matrix_norm(delta, ord=2))}

    return {
        "surrogate": norms(surrogate_delta, c_dir.double()),
        "projection": norms(corrected_delta, corrected_dir),
        "gram": correction.as_metrics(),
        "interface_attribution": cumulative_interface_attribution(edges, c_dir, correction),
        "calibration_samples": int(calibration.chain[0].shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", default="product_endpoint")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", required=True,
                        choices=["random", "collapsed", "convergence", "convergence_extended"])
    parser.add_argument("--collapse-scale", type=float, default=0.01)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--gram-corrected", action="store_true",
                        help="report the projection defect alongside the surrogate")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    config = json.loads(
        (Path(arguments.config_dir) / f"gate1_cifar10_{VARIANT_TAGS[arguments.variant]}.json").read_text()
    )
    config["seed"] = arguments.seed
    L.seed_everything(arguments.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = GateDataModule(config["data"], arguments.seed)
    data_module.setup()
    module = HierarchyCertificateModule(config)

    if arguments.mode != "random":
        unit = Path(arguments.output_root) / "units" / f"{arguments.variant}__seed{arguments.seed}"
        checkpoint = unit / "checkpoints" / "last.ckpt"
        if not checkpoint.is_file():
            raise SystemExit(f"no checkpoint at {checkpoint}")
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        module.load_state_dict(payload["state_dict"])
    # random mode deliberately keeps the fresh initialization

    module = module.to(device).eval()
    fingerprint = float(sum(float(p.detach().double().abs().sum())
                            for p in module.backbone.parameters()))
    record = {"mode": arguments.mode, "variant": arguments.variant, "seed": arguments.seed,
              "device": str(device), "ridge": arguments.ridge,
              "backbone_fingerprint": fingerprint,
              "weights_loaded": arguments.mode != "random"}
    print(f"backbone fingerprint {fingerprint:.6f} "
          f"(loaded={record['weights_loaded']})")

    if arguments.mode == "collapsed":
        module.backbone = CollapsedBackbone(module.backbone, arguments.collapse_scale).to(device)
        record["collapse_scale"] = arguments.collapse_scale

    if arguments.mode == "convergence_extended":
        # The test split is held out from SSL training exactly as the
        # calibration and validation splits are, so Stage-B may draw on
        # half of it provided Stage-C evaluates on the other half.  One
        # further doubling, no training.
        from torch.utils.data import ConcatDataset, Subset

        test = data_module.datasets["test"]
        half = len(test) // 2
        data_module.datasets["calibration"] = ConcatDataset([
            data_module.datasets["calibration"],
            data_module.datasets["val"],
            Subset(test, list(range(half))),
        ])
        data_module.datasets["test"] = Subset(test, list(range(half, len(test))))
        record["stage_b_pool"] = len(data_module.datasets["calibration"])
        record["stage_c_size"] = len(data_module.datasets["test"])

    if arguments.mode.startswith("convergence"):
        if arguments.mode == "convergence_extended":
            pool = record["stage_b_pool"]
        else:
            pool = int(config["data"]["n_calibration"]) + int(config["data"]["n_val"])
        sizes = [n for n in (625, 1250, 2500, 5000, 10000, 20000) if n <= pool]
        record["clean_pool"] = pool
        record["curve"] = {}
        for size in sizes:
            evaluation = certificate_evaluation(
                module, data_module, device, arguments.seed,
                measurement_ridge=arguments.ridge,
                pool_val_into_calibration=(arguments.mode == "convergence"),
                calibration_limit=size,
            )
            entry = summarize(evaluation)
            if arguments.gram_corrected:
                entry["gram"] = gram_corrected_report(
                    module, data_module, device, arguments.seed, arguments.ridge,
                    calibration_limit=size,
                    pool_val=(arguments.mode == "convergence"),
                )
            record["curve"][str(size)] = entry
            print(f"N={size:5d}  " + json.dumps(record["curve"][str(size)]))
        keys = sorted(record["curve"], key=int)
        record["successive_change"] = {
            f"{a}->{b}": abs(record["curve"][b]["normalized_closure_defect"]
                             - record["curve"][a]["normalized_closure_defect"])
            for a, b in zip(keys, keys[1:])
        }
        if arguments.gram_corrected:
            record["successive_change_projection"] = {
                f"{a}->{b}": abs(record["curve"][b]["gram"]["projection"]["ratio"]
                                 - record["curve"][a]["gram"]["projection"]["ratio"])
                for a, b in zip(keys, keys[1:])
            }
            print("projection successive change:",
                  json.dumps(record["successive_change_projection"]))
        print("successive change:", json.dumps(record["successive_change"]))
    else:
        evaluation = certificate_evaluation(
            module, data_module, device, arguments.seed, measurement_ridge=arguments.ridge
        )
        record.update(summarize(evaluation))
        print(json.dumps(record, indent=2))

    record["status"] = "complete"
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
