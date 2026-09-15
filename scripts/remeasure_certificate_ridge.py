"""Post-hoc Stage-C measurement-floor calibration (disclosed analysis).

Round-2 arms trained to closure 0.155 yet measured 0.300, and three
different mechanisms cluster at defect 0.296-0.303 while even flat rows
measure 0.43-0.45: the Stage-B/C whitening at ridge 1e-3 / K=128 /
calibration 2500 appears to impose an estimation-noise floor.  This
script reloads saved checkpoints and re-runs the frozen evaluation at
several measurement ridges; it changes NO frozen verdicts, it
calibrates the instrument.
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
from run_gate1_unit import VARIANT_TAGS, certificate_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ridges", default="0.001,0.01,0.1")
    parser.add_argument("--pool-val", action="store_true",
                        help="pool the val split into Stage-B calibration (5000 held-out)")
    arguments = parser.parse_args()

    unit_dir = Path(arguments.output_root) / "units" / f"{arguments.variant}__seed{arguments.seed}"
    checkpoint_path = unit_dir / "checkpoints" / "last.ckpt"
    if not checkpoint_path.is_file():
        raise SystemExit(f"no checkpoint at {checkpoint_path}")
    config = json.loads(
        (Path(arguments.config_dir) / f"gate1_cifar10_{VARIANT_TAGS[arguments.variant]}.json").read_text()
    )
    config["seed"] = arguments.seed
    L.seed_everything(arguments.seed, workers=True)
    device = torch.device("cuda")
    data_module = GateDataModule(config["data"], arguments.seed)
    data_module.setup()
    module = HierarchyCertificateModule(config)
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    module.load_state_dict(payload["state_dict"])
    module = module.to(device).eval()

    results = {}
    for ridge in [float(value) for value in arguments.ridges.split(",")]:
        evaluation = certificate_evaluation(
            module, data_module, device, arguments.seed, measurement_ridge=ridge,
            pool_val_into_calibration=arguments.pool_val,
        )
        results[str(ridge)] = {
            "normalized_closure_defect": evaluation["normalized_closure_defect"],
            "dir_frobenius": evaluation["dir_frobenius"],
            "delta_operator": evaluation["point"]["delta_operator"],
            "certified_spectrum_top": max(evaluation["point"]["certified_spectrum"]),
            "path_top": max(evaluation["point"]["path_singular_values"]),
            "endpoint_top": max(evaluation["point"]["endpoint_singular_values"]),
        }
        print(arguments.variant, "ridge", ridge, json.dumps(results[str(ridge)]))
    suffix = "_pooled" if arguments.pool_val else ""
    (unit_dir / f"remeasure_ridge{suffix}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
