"""Layer-wise profile of a trained gate unit: the theory's own predictions.

The gate reports final-layer probe and kNN only.  The path-supported
closure theory predicts effects at the INTERIOR INTERFACE, not at the
endpoint: each intermediate subspace is pushed toward being closed
under its two neighbouring operators (the (I - P_j) factors in the
Theorem-1 defect expansion), which should show up as

  - a better linear probe at the interior tap (stage 2 = layer3),
  - a HIGHER effective rank there (closure fights compression: to let
    C_comp catch a large C_dir, interior operators must stay close to
    isometric on the relevant subspace),
  - larger canonical correlations across the interior interface (more
    of the layer is handed forward rather than contracted away).

This script measures all three on saved checkpoints.  No training: it
extracts pooled per-stage features under the deterministic probe
transform, fits a convex multinomial probe per stage (LBFGS from
zeros, so the optimum -- and therefore the comparison across variants
-- is deterministic), and reports spectra.

Effective rank uses the same entropy formula as the gate's
representation_diagnostics, so stage-3 numbers are comparable to the
values already recorded in every unit.json.

Usage:
  run_layerwise_profile.py --config-dir configs/gate_v8 \
      --output-root results/gate1/gate1_20260820_v8 \
      --variant product_endpoint --seed 1
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
from run_gate1_unit import VARIANT_TAGS, _plain_loaders, dataset_classes

PROFILE_VERSION = "layerwise_profile_20260823_v1"


@torch.no_grad()
def block_taps(backbone):
    """(name, module) after the stem and after every residual block.

    Four stage-level points undersample the profile: the taps sit at stage
    boundaries, so a stage-level curve only ever shows the boundaries and
    says nothing about what happens between them.  The chain track found
    that the fine profile is a sawtooth rather than a slope, which a
    four-point curve cannot distinguish from a smooth rise.
    """

    # The gate wraps the ResNet in StageBackbone, which holds the stages in a
    # ModuleList and exposes no .layer1 attribute.  Read them from there.
    taps = [("stem", backbone.stem)]
    for stage_index, stage in enumerate(backbone.stages):
        for index, block in enumerate(stage):
            taps.append((f"layer{stage_index + 1}.{index}", block))
    return taps


@torch.no_grad()
def block_features(backbone, loader, device, max_batches=0):
    """Pooled features after the stem and after every residual block."""

    backbone.eval()
    taps = block_taps(backbone)
    collected, labels = [[] for _ in taps], []
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        value = images.to(device)
        for index, (_, module) in enumerate(taps):
            value = module(value)
            collected[index].append(backbone.pool(value).flatten(1).float().cpu())
        labels.append(targets)
    return [torch.cat(parts) for parts in collected], torch.cat(labels), [n for n, _ in taps]


@torch.no_grad()
def stage_features(backbone, loader, device, max_batches=0):
    """Pooled features after every backbone stage, plus labels.

    no_grad matters here: each appended CPU tensor otherwise keeps its
    GPU autograd graph alive, so memory grows with every batch and the
    98-batch pass over 50000 images exceeds a 40 GB card.  block_features
    already had it; this path only ever ran on larger GPUs before.
    """

    backbone.eval()
    stages, labels = None, []
    for index, (images, targets) in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        pooled = backbone.forward_stages(images.to(device))
        if stages is None:
            stages = [[] for _ in pooled]
        for index, value in enumerate(pooled):
            stages[index].append(value.float().cpu())
        labels.append(targets)
    return [torch.cat(parts) for parts in stages], torch.cat(labels)


def convex_probe(train_x, train_y, test_x, test_y, classes, device,
                 weight_decay=1e-4, iterations=200):
    """Multinomial logistic regression, LBFGS from zeros.

    The objective is convex, so starting from zeros makes the result a
    property of the features alone -- no probe seed, no schedule.
    """

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    xtr = ((train_x - mean) / std).to(device)
    xte = ((test_x - mean) / std).to(device)
    ytr, yte = train_y.to(device), test_y.to(device)

    weight = torch.zeros(xtr.shape[1], classes, device=device, requires_grad=True)
    bias = torch.zeros(classes, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=iterations, history_size=10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(xtr @ weight + bias, ytr)
        loss = loss + weight_decay * (weight * weight).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        test_accuracy = float(((xte @ weight + bias).argmax(dim=1) == yte).float().mean())
        train_accuracy = float(((xtr @ weight + bias).argmax(dim=1) == ytr).float().mean())
    return {"test_accuracy": test_accuracy, "train_accuracy": train_accuracy}


def spectrum_statistics(features):
    """Entropy effective rank -- same formula as representation_diagnostics."""

    centered = (features - features.mean(dim=0, keepdim=True)).double()
    covariance = centered.transpose(0, 1) @ centered / centered.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = float(eigenvalues.sum())
    if total <= 0:
        return {"effective_rank": 0.0, "top_eigenvalue_share": 1.0, "trace": 0.0,
                "dimension": int(features.shape[1])}
    probabilities = eigenvalues / total
    entropy = float(-(probabilities * (probabilities + 1e-20).log()).sum())
    return {
        "effective_rank": float(torch.exp(torch.tensor(entropy))),
        "normalized_effective_rank": float(torch.exp(torch.tensor(entropy))) / features.shape[1],
        "top_eigenvalue_share": float(eigenvalues.max() / total),
        "trace": total,
        "dimension": int(features.shape[1]),
    }


def canonical_correlations(lower, upper, ridge=1e-3, top_k=8):
    """Canonical correlations across an adjacent-stage interface.

    These are the contraction factors of the theory's data-processing
    argument: how much of one stage survives, linearly, into the next.
    """

    a = (lower - lower.mean(dim=0, keepdim=True)).double()
    b = (upper - upper.mean(dim=0, keepdim=True)).double()
    n = a.shape[0]
    cov_aa = a.transpose(0, 1) @ a / n
    cov_bb = b.transpose(0, 1) @ b / n
    cov_ab = a.transpose(0, 1) @ b / n
    eye_a = torch.eye(cov_aa.shape[0], dtype=torch.float64)
    eye_b = torch.eye(cov_bb.shape[0], dtype=torch.float64)
    cov_aa = cov_aa + ridge * float(cov_aa.diagonal().mean()) * eye_a
    cov_bb = cov_bb + ridge * float(cov_bb.diagonal().mean()) * eye_b

    def inverse_sqrt(matrix):
        values, vectors = torch.linalg.eigh(matrix)
        return vectors @ torch.diag(values.clamp_min(1e-12).rsqrt()) @ vectors.transpose(0, 1)

    whitened = inverse_sqrt(cov_aa) @ cov_ab @ inverse_sqrt(cov_bb)
    singular = torch.linalg.svdvals(whitened).clamp(0.0, 1.0)
    return {
        "top": float(singular[0]),
        "mean_top_k": float(singular[:top_k].mean()),
        "sum": float(singular.sum()),
        "count_above_half": int((singular > 0.5).sum()),
    }


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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="encode only this many batches per split (0 = all). For "
                             "correctness smoke runs: the output is then written to a "
                             "*_smoke.json name and marked truncated, so a partial "
                             "profile can never be mistaken for a real one.")
    parser.add_argument("--granularity", choices=("stage", "block"), default="stage",
                        help="stage = 4 taps at stage boundaries; block = stem plus "
                             "every residual block, which is what shows the shape "
                             "between boundaries")
    parser.add_argument("--force", action="store_true",
                        help="recompute a profile that already exists")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="opt in to CPU; the encoder is far slower there")
    parser.add_argument("--checkpoint", default="",
                        help="evaluate this checkpoint instead of last.ckpt; "
                             "a milestone checkpoint writes its own output name, "
                             "so epoch 20 can never overwrite epoch 800")
    parser.add_argument("--probe-subsample", type=int, default=0,
                        help="use only this many training points for the probes (0 = all)")
    arguments = parser.parse_args()

    unit_dir = Path(arguments.output_root) / "units" / f"{arguments.variant}__seed{arguments.seed}"
    # Idempotency from the target state, not from a submitter's ledger.
    out_name = ("layerwise_profile.json" if arguments.granularity == "stage"
                else "blockwise_profile.json")
    if arguments.checkpoint:
        # e.g. epoch-0020.ckpt -> blockwise_profile_epoch-0020.json
        stem = Path(arguments.checkpoint).stem
        if stem != "last":
            out_name = out_name.replace(".json", f"_{stem}.json")
    if arguments.max_batches:
        # A truncated run never claims the canonical filename.  This is a
        # correctness check, not a measurement, and the two must not be
        # confusable on disk.
        out_name = out_name.replace(".json", "_smoke.json")
    if (unit_dir / out_name).is_file() and not arguments.force:
        print(f"{out_name} already exists for {arguments.variant} "
              f"seed{arguments.seed}; pass --force to recompute")
        return
    checkpoint_path = (Path(arguments.checkpoint) if arguments.checkpoint
                       else unit_dir / "checkpoints" / "last.ckpt")
    if not checkpoint_path.is_file():
        raise SystemExit(f"no checkpoint at {checkpoint_path}")
    config = json.loads(
        (Path(arguments.config_dir) / f"gate1_cifar10_{VARIANT_TAGS[arguments.variant]}.json").read_text()
    )
    config["seed"] = arguments.seed
    L.seed_everything(arguments.seed, workers=True)
    device = require_accelerator(arguments.allow_cpu)

    module = HierarchyCertificateModule(config)
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    module.load_state_dict(payload["state_dict"])
    module = module.to(device).eval()

    train_loader, test_loader = _plain_loaders(config["data"])
    if arguments.granularity == "block":
        train_features, train_labels, tap_names = block_features(
            module.backbone, train_loader, device, arguments.max_batches)
        test_features, test_labels, _ = block_features(
            module.backbone, test_loader, device, arguments.max_batches)
    else:
        train_features, train_labels = stage_features(
            module.backbone, train_loader, device, arguments.max_batches)
        test_features, test_labels = stage_features(
            module.backbone, test_loader, device, arguments.max_batches)
        tap_names = [f"layer{i + 1}" for i in range(len(train_features))]
    classes = dataset_classes(config)

    if arguments.probe_subsample:
        keep = torch.randperm(train_labels.shape[0])[: arguments.probe_subsample]
        train_features = [f[keep] for f in train_features]
        train_labels = train_labels[keep]

    # Which taps the hierarchy actually reads, derived from level_stages rather
    # than hard-coded to stage indices: at block granularity a fixed map would
    # label the wrong taps, and level_stages is the only source of truth.
    level_stages = list(config["model"]["level_stages"])
    role_of_stage = {}
    for level, stage_index in enumerate(level_stages):
        role_of_stage[stage_index] = (
            "root tap (level 0)" if level == 0 else
            f"endpoint tap (level {level})" if level == len(level_stages) - 1 else
            f"interior interface (level {level})")
    tap_roles = []
    for index, name in enumerate(tap_names):
        stage_index = (index - 1 if arguments.granularity == "stage"
                       else (-1 if name == "stem" else int(name[5]) - 1))
        role = role_of_stage.get(stage_index, "below/between taps")
        last_block = len(module.backbone.stages[stage_index]) - 1 if stage_index >= 0 else -1
        if arguments.granularity == "block" and role != "below/between taps" \
                and not name.endswith(f".{last_block}"):
            role = f"inside {role.split(' (')[0]} stage"
        tap_roles.append(role)

    stages = []
    for index, (train_x, test_x) in enumerate(zip(train_features, test_features)):
        probe = convex_probe(train_x, train_labels, test_x, test_labels, classes, device)
        statistics = spectrum_statistics(test_x)
        stages.append({
            "stage": index,
            "backbone_layer": tap_names[index],
            "level_role": tap_roles[index],
            "probe": probe,
            "spectrum": statistics,
        })
        print(f"stage {index}: probe={probe['test_accuracy']*100:.2f}% "
              f"eff_rank={statistics['effective_rank']:.1f}/{statistics['dimension']}")

    interfaces = []
    for index in range(len(test_features) - 1):
        interfaces.append({
            "interface": f"stage{index}->stage{index + 1}",
            "canonical_correlations": canonical_correlations(
                test_features[index], test_features[index + 1]
            ),
        })

    record = {
        "profile_version": PROFILE_VERSION,
        "checkpoint": str(checkpoint_path),
        "variant": arguments.variant,
        "seed": arguments.seed,
        "dataset": config["data"]["dataset"],
        "level_stages": config["model"]["level_stages"],
        "stages": stages,
        "interfaces": interfaces,
        "status": "complete",
    }
    record["granularity"] = arguments.granularity
    record["truncated_smoke_run"] = bool(arguments.max_batches)
    if arguments.max_batches:
        record["max_batches"] = arguments.max_batches
        record["status"] = "smoke"
    (unit_dir / out_name).write_text(json.dumps(record, indent=2))
    print(json.dumps({"variant": arguments.variant, "seed": arguments.seed,
                      "probe_by_stage": [round(s["probe"]["test_accuracy"] * 100, 2) for s in stages],
                      "eff_rank_by_stage": [round(s["spectrum"]["effective_rank"], 1) for s in stages]},
                     indent=2))


if __name__ == "__main__":
    main()
