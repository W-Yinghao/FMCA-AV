"""Q2: per-stage probe curve for an EXTERNAL SSL baseline.

The comparison this exists to make is "our stage k against their stage
k".  It is legitimate here because the baseline stack and the gate
stack instantiate the same backbone class (fmca_av/resnet.py
CIFARResNet, width 64), so the stages being compared are the same
objects; this script only reads them, it changes no network.

Probe protocol is copied from run_layerwise_profile so the two sides
are scored identically: pooled features after each stage under the
deterministic probe transform, convex multinomial probe (LBFGS from
zeros, so there is no probe seed), same weight decay, same iterations.

Spectrum diagnostics ride along because a probe number alone cannot
distinguish "informative" from "one direction carries the label".
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmca_av.baselines import BaselineSSL
from run_gate1_unit import _plain_loaders, dataset_classes
from run_layerwise_profile import convex_probe, spectrum_statistics

STAGE_NAMES = ["layer1", "layer2", "layer3", "layer4"]


@torch.no_grad()
def stage_features(backbone, loader, device):
    """Pooled features after each of the four stages, plus labels."""

    backbone.eval()
    collected = [[] for _ in STAGE_NAMES]
    labels = []
    for images, targets in loader:
        values = backbone.stem(images.to(device))
        for index, name in enumerate(STAGE_NAMES):
            values = getattr(backbone, name)(values)
            collected[index].append(backbone.pool(values).flatten(1).float().cpu())
        labels.append(targets)
    return [torch.cat(parts) for parts in collected], torch.cat(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--probe-subsample", type=int, default=0)
    arguments = parser.parse_args()

    config = json.loads(Path(arguments.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module = BaselineSSL(config)
    payload = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = module.load_state_dict(payload["state_dict"], strict=False)
    # Loud, not silent: a checkpoint whose backbone did not land is a
    # random encoder wearing a method's name -- the exact failure the
    # validity gate caught once already.
    backbone_missing = [k for k in missing if k.startswith("backbone.")]
    if backbone_missing:
        raise SystemExit(f"checkpoint did not supply {len(backbone_missing)} backbone tensors")
    backbone = module.backbone.to(device).eval()
    fingerprint = float(sum(float(p.detach().double().abs().sum()) for p in backbone.parameters()))
    print(f"backbone fingerprint {fingerprint:.6f} (unexpected keys: {len(unexpected)})")

    train_loader, test_loader = _plain_loaders(config["data"])
    train_features, train_labels = stage_features(backbone, train_loader, device)
    test_features, test_labels = stage_features(backbone, test_loader, device)
    classes = dataset_classes(config)

    if arguments.probe_subsample:
        limit = arguments.probe_subsample
        train_features = [f[:limit] for f in train_features]
        train_labels = train_labels[:limit]

    stages = []
    for index, name in enumerate(STAGE_NAMES):
        probe = convex_probe(train_features[index], train_labels,
                             test_features[index], test_labels, classes, device)
        stages.append({"stage": index, "backbone_layer": name, "probe": probe,
                       "spectrum": spectrum_statistics(test_features[index])})
        print(f"  {name}: {probe['test_accuracy']*100:.2f}%  "
              f"eff_rank {stages[-1]['spectrum']['effective_rank']:.1f}")

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "profile_version": "baseline_layerwise_v1",
        "method": config["experiment"].get("method"),
        "seed": config.get("seed"),
        "dataset": config["data"].get("dataset"),
        "backbone_fingerprint": fingerprint,
        "checkpoint": str(arguments.checkpoint),
        "stages": stages, "status": "complete",
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
