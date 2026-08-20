"""Plug-in study unit runner: one (method, row, seed).

Reuses the gate runner's training/evaluation machinery with the
PluginSSLModule.  Usage:
  run_plugin_unit.py --config configs/plugin/<file>.json --seed N [--probe-mode]
"""

import argparse
import json
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from fmca_av.certificate.gate_data import GateDataModule
from fmca_av.certificate.plugin_module import PluginSSLModule
from fmca_av.certificate.triplet import CERTIFICATE_VERSION
from fmca_av.knn import weighted_knn_accuracy
from run_gate1_unit import (
    DivergenceGuard,
    _plain_loaders,
    certificate_evaluation,
    linear_probe_evaluation,
    representation_diagnostics,
    resumable_checkpoint,
)

PLUGIN_VERSION = "plugin_20260820_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", default="results/plugin/" + PLUGIN_VERSION)
    parser.add_argument("--probe-mode", action="store_true")
    arguments = parser.parse_args()

    config = json.loads(Path(arguments.config).read_text())
    config["seed"] = arguments.seed
    row = f"{config['base']['name']}_{'plugin' if config.get('plugin', {}).get('enabled') else 'base'}"
    unit_key = f"{row}/seed{arguments.seed}"
    subdir = "probe" if arguments.probe_mode else "units"
    unit_dir = Path(arguments.output_root) / subdir / f"{row}__seed{arguments.seed}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    record_path = unit_dir / "unit.json"
    if record_path.is_file() and json.loads(record_path.read_text()).get("status") == "complete":
        print(f"skip complete unit {unit_key}")
        return

    started = time.time()
    try:
        L.seed_everything(arguments.seed, workers=True)
        device = torch.device("cuda")
        data_module = GateDataModule(config["data"], arguments.seed)
        data_module.setup()
        module = PluginSSLModule(config)
        max_epochs = 2 if arguments.probe_mode else int(config["trainer"]["max_epochs"])
        trainer = L.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=max_epochs,
            deterministic="warn",
            check_val_every_n_epoch=1 if arguments.probe_mode else 10,
            callbacks=[
                ModelCheckpoint(dirpath=str(unit_dir / "checkpoints"), save_last=True,
                                save_top_k=0, every_n_epochs=1),
                DivergenceGuard(),
            ],
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
            "plugin_version": PLUGIN_VERSION,
            "certificate_version": CERTIFICATE_VERSION,
            "base": config["base"]["name"],
            "plugin_enabled": bool(config.get("plugin", {}).get("enabled")),
            "seed": arguments.seed,
            "probe_mode": arguments.probe_mode,
            "train_epochs": max_epochs,
            "certificate": certificate,
            "linear_probe": probe_metrics,
            "knn_accuracy": float(knn_accuracy),
            "representation": diagnostics,
            "collapsed": bool(probe_metrics["test_accuracy"] < 0.15),
            "wall_seconds": time.time() - started,
            "gpu": torch.cuda.get_device_name(0),
            "status": "complete",
        }
    except Exception as error:
        record = {"unit_key": unit_key, "status": "failed", "reason": repr(error),
                  "wall_seconds": time.time() - started}
        record_path.write_text(json.dumps(record, indent=2))
        raise
    record_path.write_text(json.dumps(record, indent=2))
    print(json.dumps({"unit_key": unit_key, "status": record["status"],
                      "probe": record["linear_probe"]["test_accuracy"],
                      "knn": record["knn_accuracy"],
                      "defect": record["certificate"]["normalized_closure_defect"]}, indent=2))


if __name__ == "__main__":
    main()
