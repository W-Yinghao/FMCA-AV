"""Channel spectroscopy: per-edge operator spectra as a function of the
edge channel's strength, measured post-hoc on a fixed checkpoint.

The designed-channel superpower made measurable: sweep one channel
parameter of one edge (crop scale floor, jitter strength, grayscale
probability), rebuild the view tree, and re-run the frozen Stage-B/C
evaluation on a FROZEN model.  Output: sigma spectra of every edge plus
the triplet quantities per sweep point -- the instrument's response
curve to channel design.

Usage:
  run_channel_spectroscopy.py --config <train config> --checkpoint <last.ckpt>
      --edge 0 --parameter min_scale --values 0.1,0.2,0.4,0.6,0.8
      --out <output json>
"""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightning as L

from fmca_av.certificate.gate_data import GateDataModule
from fmca_av.certificate.hierarchy_module import HierarchyCertificateModule
from run_gate1_unit import certificate_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--edge", type=int, required=True)
    parser.add_argument("--parameter", required=True,
                        help="edge spec field to sweep (e.g. min_scale, color_jitter_strength, grayscale_probability, patch_fraction)")
    parser.add_argument("--values", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    config = json.loads(Path(arguments.config).read_text())
    config["seed"] = arguments.seed
    L.seed_everything(arguments.seed, workers=True)
    device = torch.device("cuda")
    module = HierarchyCertificateModule(config)
    payload = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    module.load_state_dict(payload["state_dict"], strict=False)
    module = module.to(device).eval()

    sweep = {}
    for value in (float(v) for v in arguments.values.split(",")):
        swept = json.loads(json.dumps(config))
        edge = swept["data"]["view_tree"]["edges"][arguments.edge]
        edge[arguments.parameter] = value
        data_module = GateDataModule(swept["data"], arguments.seed)
        data_module.setup()
        evaluation = certificate_evaluation(module, data_module, device, arguments.seed)
        point = evaluation["point"]
        sweep[str(value)] = {
            "edge_top_singular_values": point["edge_top_singular_values"],
            "edge_frobenius_norms": point["edge_frobenius_norms"],
            "endpoint_singular_values": point["endpoint_singular_values"][:16],
            "path_singular_values": point["path_singular_values"][:16],
            "normalized_closure_defect": evaluation["normalized_closure_defect"],
            "certified_spectrum_top": max(point["certified_spectrum"]),
        }
        print(arguments.parameter, value, "->",
              {k: sweep[str(value)][k] for k in ("edge_frobenius_norms", "normalized_closure_defect")})
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": arguments.config,
        "checkpoint": arguments.checkpoint,
        "edge": arguments.edge,
        "parameter": arguments.parameter,
        "sweep": sweep,
    }, indent=2))


if __name__ == "__main__":
    main()
