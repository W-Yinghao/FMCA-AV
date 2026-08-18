"""Gate 1 unit runner: one (variant, seed) of the CIFAR-10 structure gate.

Prereg: prereg/GATE1_CIFAR10_STRUCTURE_PREREG_FROZEN_20260816.md
Pipeline per unit: train HierarchyCertificateModule on the nested view
tree -> Stage-B coordinates on the calibration split -> Stage-C
certificate on the test split (point + two-fold cross-fit + controls)
-> frozen linear probe + weighted kNN + collapse diagnostics.

Units are addressed by real keys (variant/seed), write one JSON each,
skip when complete, resume training from last.ckpt after any kill, and
record failures loud with a reason code.
"""

import argparse
import json
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


class DivergenceGuard(Callback):
    """Abort loudly if the bounded objective ever leaves its sane range.

    With whitened operators every score term is <= ~1 per mode, so a
    score far beyond that bound means the optimizer found a degenerate
    channel again (v1: scale runaway to -1e19; v2 probe: whitening
    estimator gaming to endpoint score 6.96).  Failing here fails the
    probe job, which blocks the dependent fleet automatically.
    """

    SCORE_BOUNDS = {
        "train/endpoint_score": 2.5,
        "train/leaf_score": 2.5,
        "train/edge_score_sum": 5.0,
        "train/cross_score_sum": 7.5,
        "train/product_score": 2.5,
        # The faithful flat trace is bounded by K (=128 modes), not by 1.
        "train/flat_trace_score": 200.0,
        "train/loss": 300.0,
    }

    def on_train_epoch_end(self, trainer, module) -> None:
        for name, bound in self.SCORE_BOUNDS.items():
            value = trainer.callback_metrics.get(name)
            if value is None:
                continue
            if not torch.isfinite(value) or abs(float(value)) > bound:
                raise RuntimeError(
                    f"divergence guard: {name}={float(value):.3e} exceeds bound {bound}"
                )
from torch.utils.data import DataLoader

from fmca_av.certificate.controls import (
    endpoint_pairing_shuffle,
    pairing_noise_floor,
    random_orthogonal,
    rotate_interface,
)
from fmca_av.certificate.coordinates import fit_level_coordinates
from fmca_av.certificate.estimation import (
    ChainFeatureBatch,
    crossfit_edge_and_endpoint,
    encode_chain_batch,
    estimate_edge_operators,
    estimate_endpoint_operator,
    level_calibration_features,
)
from fmca_av.certificate.gate_data import GateDataModule
from fmca_av.certificate.hierarchy_module import GATE_VARIANTS, HierarchyCertificateModule
from fmca_av.certificate.triplet import (
    CERTIFICATE_VERSION,
    certificate_report,
    compose_edge_operators,
)
from fmca_av.data.cifar import CIFARFiles, CIFARProbeTransform, LabeledCIFARDataset
from fmca_av.knn import weighted_knn_accuracy
from fmca_av.probe_module import LinearProbe

GATE_VERSION = "gate1_20260817_v3"
VARIANT_TAGS = {
    "final_2view": "v1_final_2view",
    "final_mview": "v2_final_mview",
    "additive_2view": "v3_additive_2view",
    "additive_mview": "v4_additive_mview",
    "amdim_cross": "v5_amdim_cross",
    "product_only": "v6_product_only",
    "product_endpoint": "v7_product_endpoint",
}


def resumable_checkpoint(unit_dir: Path) -> "str | None":
    """Return last.ckpt only if it is loadable with finite weights.

    A NaN-poisoned or format-incompatible checkpoint (e.g. saved by a
    pre-stability-fix scheduler) is quarantined loudly and training starts
    fresh; silently resuming poisoned state manufactures fake results.
    """

    last = unit_dir / "checkpoints" / "last.ckpt"
    if not last.is_file():
        return None
    try:
        payload = torch.load(str(last), map_location="cpu", weights_only=False)
        for name, value in payload["state_dict"].items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"non-finite weights in {name}")
        for state in payload.get("lr_schedulers", []):
            # The warmup+cosine SequentialLR stores child states under this
            # key; a checkpoint from any other scheduler cannot restore.
            if "_schedulers" not in state:
                raise ValueError("lr scheduler state predates the warmup scheduler")
        return str(last)
    except Exception as error:
        quarantine = last.with_name("last.quarantined.ckpt")
        last.rename(quarantine)
        print(f"QUARANTINED incompatible checkpoint ({error!r}); starting fresh")
        return None


def _move_batch(batch, device):
    return {
        "chain": [images.to(device) for images in batch["chain"]],
        "children": [images.to(device) for images in batch["children"]],
        "endpoint": batch["endpoint"].to(device),
    }


@torch.no_grad()
def collect_chain_features(module, loader, device, max_parents):
    module.eval()
    chains, childrens, endpoints = None, None, []
    collected = 0
    for batch in loader:
        features = module.feature_batch(_move_batch(batch, device))
        if chains is None:
            chains = [[] for _ in features.chain]
            childrens = [[] for _ in features.children]
        for level, states in enumerate(features.chain):
            chains[level].append(states.double().cpu())
        for edge, descendants in enumerate(features.children):
            childrens[edge].append(descendants.double().cpu())
        endpoints.append(features.endpoint_descendants.double().cpu())
        collected += features.chain[0].shape[0]
        if max_parents and collected >= max_parents:
            break
    return ChainFeatureBatch(
        chain=[torch.cat(parts)[:max_parents or None] for parts in chains],
        children=[torch.cat(parts)[:max_parents or None] for parts in childrens],
        endpoint_descendants=torch.cat(endpoints)[:max_parents or None],
    )


def certificate_evaluation(module, data_module, device, seed):
    calibration = collect_chain_features(
        module, data_module.calibration_dataloader(), device, max_parents=0
    )
    coordinates = [
        fit_level_coordinates(level_calibration_features(calibration, level))
        for level in range(calibration.num_levels)
    ]
    raw = collect_chain_features(module, data_module.test_dataloader(), device, max_parents=0)
    encoded = encode_chain_batch(raw, coordinates)
    edges = estimate_edge_operators(encoded)
    c_dir = estimate_endpoint_operator(encoded)
    point = certificate_report(c_dir, edges=edges)
    folds = crossfit_edge_and_endpoint(encoded, torch.Generator().manual_seed(seed + 71))
    fold_reports = [
        certificate_report(fold_dir, edges=fold_edges).as_metrics()
        for fold_edges, fold_dir in folds
    ]
    if len(edges) >= 2:
        rotation = random_orthogonal(
            edges[0].shape[1], torch.Generator().manual_seed(seed + 72)
        )
        base_comp = compose_edge_operators(edges)
        gauge_change = float(
            (compose_edge_operators(rotate_interface(edges, 1, rotation, "both")) - base_comp)
            .abs()
            .max()
        )
        one_sided_change = float(
            (compose_edge_operators(rotate_interface(edges, 1, rotation, "left")) - base_comp)
            .abs()
            .max()
        )
    else:
        # A single-edge chain has no interior interface to rotate.
        gauge_change = None
        one_sided_change = None
    pairing_floor = pairing_noise_floor(
        encoded.chain[0], encoded.children[0], repeats=10,
        generator=torch.Generator().manual_seed(seed + 73),
    )
    endpoint_floor = torch.linalg.matrix_norm(
        endpoint_pairing_shuffle(encoded, torch.Generator().manual_seed(seed + 74)), ord="fro"
    )
    dir_norm = float(torch.linalg.matrix_norm(c_dir, ord="fro"))
    return {
        "point": point.as_metrics(),
        "normalized_closure_defect": point.delta_frobenius / (dir_norm + 1e-12),
        "dir_frobenius": dir_norm,
        "crossfit_folds": fold_reports,
        "controls": {
            "gauge_invariance_max_change": gauge_change,
            "one_sided_rotation_max_change": one_sided_change,
            "pairing_floor_max": float(pairing_floor.max()),
            "first_edge_frobenius": float(torch.linalg.matrix_norm(edges[0], ord="fro")),
            "endpoint_shuffle_frobenius": float(endpoint_floor),
        },
    }


def _plain_loaders(data_config, batch_size=512, workers=4):
    root, dataset = str(data_config["root"]), str(data_config["dataset"])
    transform = CIFARProbeTransform(False, [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    train = LabeledCIFARDataset(CIFARFiles(root, dataset, train=True), transform)
    test = LabeledCIFARDataset(CIFARFiles(root, dataset, train=False), transform)
    return (
        DataLoader(train, batch_size=batch_size, num_workers=workers, shuffle=False),
        DataLoader(test, batch_size=batch_size, num_workers=workers, shuffle=False),
    )


@torch.no_grad()
def representation_diagnostics(backbone, loader, device):
    backbone.eval()
    features = []
    for images, _ in loader:
        features.append(backbone(images.to(device)).float().cpu())
    stacked = torch.cat(features)
    centered = stacked - stacked.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered / centered.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(0)
    total = float(eigenvalues.sum())
    if total <= 0:
        return {"effective_rank": 0.0, "top_eigenvalue_share": 1.0, "trace": 0.0}
    probabilities = eigenvalues / total
    entropy = float(-(probabilities * (probabilities + 1e-20).log()).sum())
    return {
        "effective_rank": float(torch.exp(torch.tensor(entropy))),
        "top_eigenvalue_share": float(eigenvalues.max() / total),
        "trace": total,
    }


def linear_probe_evaluation(module, config, seed, unit_dir, probe_mode):
    from fmca_av.data.cifar import CIFARProbeDataModule

    probe_config = dict(config["probe"])
    if probe_mode:
        probe_config["epochs"] = 3
    probe_data = CIFARProbeDataModule(config["data"], probe_config, seed)
    probe_data.setup()
    probe = LinearProbe(
        module.backbone, module.backbone.output_dim, 10, probe_config
    )
    probe.probe_config["max_epochs"] = int(probe_config.get("epochs", 100))
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=int(probe_config.get("epochs", 100)),
        deterministic="warn",
        logger=CSVLogger(str(unit_dir), name="probe_logs"),
        enable_progress_bar=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(probe, probe_data.train_dataloader(), probe_data.val_dataloader())
    results = trainer.test(probe, probe_data.test_dataloader())
    return {
        "test_accuracy": float(results[0]["test/accuracy"]),
        "test_top5_accuracy": float(results[0]["test/top5_accuracy"]),
        "epochs": int(probe_config.get("epochs", 100)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(GATE_VARIANTS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config-dir", default="configs/gate")
    parser.add_argument("--output-root", default="results/gate1/" + GATE_VERSION)
    parser.add_argument("--probe-mode", action="store_true", help="2-epoch QC probe of the full pipeline")
    arguments = parser.parse_args()

    unit_key = f"{arguments.variant}/seed{arguments.seed}"
    unit_dir = Path(arguments.output_root) / "units" / f"{arguments.variant}__seed{arguments.seed}"
    if arguments.probe_mode:
        unit_dir = Path(arguments.output_root) / "probe" / f"{arguments.variant}__seed{arguments.seed}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    record_path = unit_dir / "unit.json"
    if record_path.is_file() and json.loads(record_path.read_text()).get("status") == "complete":
        print(f"skip complete unit {unit_key}")
        return

    config_path = Path(arguments.config_dir) / f"gate1_cifar10_{VARIANT_TAGS[arguments.variant]}.json"
    config = json.loads(config_path.read_text())
    config["seed"] = arguments.seed
    started = time.time()
    try:
        L.seed_everything(arguments.seed, workers=True)
        device = torch.device("cuda")
        data_module = GateDataModule(config["data"], arguments.seed)
        data_module.setup()
        module = HierarchyCertificateModule(config)
        max_epochs = 2 if arguments.probe_mode else int(config["trainer"]["max_epochs"])
        checkpoint = ModelCheckpoint(
            dirpath=str(unit_dir / "checkpoints"), save_last=True, save_top_k=0,
            every_n_epochs=1,
        )
        guard = DivergenceGuard()
        trainer = L.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=max_epochs,
            deterministic="warn",
            check_val_every_n_epoch=1 if arguments.probe_mode else 10,
            callbacks=[checkpoint, guard],
            logger=CSVLogger(str(unit_dir), name="train_logs"),
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            gradient_clip_val=1.0,
        )
        trainer.fit(
            module,
            data_module.train_dataloader(),
            data_module.val_dataloader(),
            ckpt_path=resumable_checkpoint(unit_dir),
        )
        module = module.to(device)
        certificate = certificate_evaluation(module, data_module, device, arguments.seed)
        train_loader, test_loader = _plain_loaders(config["data"])
        diagnostics = representation_diagnostics(module.backbone, test_loader, device)
        knn_accuracy = weighted_knn_accuracy(
            module.backbone, train_loader, test_loader, classes=10, device=device
        )
        probe_metrics = linear_probe_evaluation(
            module, config, arguments.seed, unit_dir, arguments.probe_mode
        )
        record = {
            "unit_key": unit_key,
            "gate_version": Path(arguments.output_root).name,
            "certificate_version": CERTIFICATE_VERSION,
            "variant": arguments.variant,
            "seed": arguments.seed,
            "probe_mode": arguments.probe_mode,
            "train_epochs": max_epochs,
            "certificate": certificate,
            "linear_probe": probe_metrics,
            "knn_accuracy": float(knn_accuracy),
            "representation": diagnostics,
            # Entropy-based effective rank sits near 1 even for healthy
            # CIFAR backbones in this repo (83% FastSSL model: 1.055), so
            # it stays a reported diagnostic; collapse gates on the probe.
            "collapsed": bool(probe_metrics["test_accuracy"] < 0.15),
            "wall_seconds": time.time() - started,
            "gpu": torch.cuda.get_device_name(0),
            "status": "complete",
        }
    except Exception as error:
        record = {
            "unit_key": unit_key,
            "status": "failed",
            "reason": repr(error),
            "wall_seconds": time.time() - started,
        }
        record_path.write_text(json.dumps(record, indent=2))
        raise
    record_path.write_text(json.dumps(record, indent=2))
    print(json.dumps({"unit_key": unit_key, "status": record["status"],
                      "knn": record.get("knn_accuracy"),
                      "probe": record.get("linear_probe", {}).get("test_accuracy"),
                      "defect": record.get("certificate", {}).get("normalized_closure_defect")},
                     indent=2))


if __name__ == "__main__":
    main()
