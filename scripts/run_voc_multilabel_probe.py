#!/usr/bin/env python3
"""Lightning frozen-backbone VOC2007 multi-label transfer evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from PIL import Image
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from fmca_av.config import load_config
from fmca_av.baselines import BaselineSSL
from fmca_av.data.imagenet import ImageNetProbeTransform
from fmca_av.vision_module import VisionFMCAAV
from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


class VOCMultiLabel(Dataset):
    def __init__(self, root: str, split: str, train: bool, size: int = 224) -> None:
        self.root = Path(root)
        split_file = self.root / "ImageSets" / "Main" / f"{split}.txt"
        self.ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.class_to_index = {name: index for index, name in enumerate(VOC_CLASSES)}
        self.transform = ImageNetProbeTransform(
            train,
            {"size": size, "eval_resize": int(round(size * 256 / 224))},
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_id = self.ids[index]
        with Image.open(self.root / "JPEGImages" / f"{image_id}.jpg") as image:
            inputs = self.transform(image.convert("RGB"))
        target = torch.zeros(len(VOC_CLASSES), dtype=torch.float32)
        tree = ET.parse(self.root / "Annotations" / f"{image_id}.xml")
        for item in tree.findall("object"):
            name = item.findtext("name", default="")
            if name in self.class_to_index:
                target[self.class_to_index[name]] = 1.0
        return inputs, target


def average_precision(scores: Tensor, targets: Tensor) -> Tensor:
    values = []
    for class_index in range(targets.shape[1]):
        labels = targets[:, class_index]
        positives = labels.sum()
        if positives == 0:
            values.append(torch.tensor(float("nan")))
            continue
        order = torch.argsort(scores[:, class_index], descending=True)
        ordered = labels[order]
        precision = ordered.cumsum(0) / torch.arange(1, len(ordered) + 1)
        values.append((precision * ordered).sum() / positives)
    return torch.stack(values)


class VOCProbe(L.LightningModule):
    def __init__(self, backbone: nn.Module, feature_dim: int, learning_rate: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, len(VOC_CLASSES))
        self.learning_rate = learning_rate
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self._validation = []
        self._test = []

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, inputs: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.backbone(inputs)
        return self.classifier(features)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        logits = self(batch[0])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch[1])
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        logits = self(batch[0])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch[1])
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self._validation.append((logits.detach().sigmoid().cpu(), batch[1].detach().cpu()))

    def on_validation_epoch_end(self) -> None:
        scores = torch.cat([item[0] for item in self._validation])
        targets = torch.cat([item[1] for item in self._validation])
        mean_ap = torch.nanmean(average_precision(scores, targets))
        self.log("val_map", mean_ap, prog_bar=True)
        self._validation.clear()

    def test_step(self, batch, batch_idx: int) -> None:
        self._test.append((self(batch[0]).detach().sigmoid().cpu(), batch[1].detach().cpu()))

    def on_test_epoch_end(self) -> None:
        scores = torch.cat([item[0] for item in self._test])
        targets = torch.cat([item[1] for item in self._test])
        per_class = average_precision(scores, targets)
        self.log("test_map", torch.nanmean(per_class))
        for index, value in enumerate(per_class):
            self.log(f"test_ap_{VOC_CLASSES[index]}", value)
        self._test.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.learning_rate, momentum=0.9)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--model-type", choices=("fmca", "baseline"), default="fmca")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    args = parser.parse_args()
    config = load_config(args.config)
    L.seed_everything(int(config["seed"]) + 9000, workers=True)
    source = (BaselineSSL.load_from_checkpoint(args.checkpoint, config=config, map_location="cpu")
              if args.model_type == "baseline" else
              VisionFMCAAV.load_from_checkpoint(args.checkpoint, config=config, map_location="cpu"))
    model = VOCProbe(source.backbone, source.backbone.output_dim, args.learning_rate)
    root = str(Path(args.root) / "VOC2007")
    train = VOCMultiLabel(root, "train", True)
    validation = VOCMultiLabel(root, "val", False)
    test = VOCMultiLabel(root, "test", False)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True)
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(dirpath=output / "checkpoints", monitor="val_map", mode="max", save_last=True)
    trainer = L.Trainer(
        accelerator="gpu", devices=1, max_epochs=args.epochs, precision="32-true", deterministic=True,
        logger=CSVLogger(str(output), name="lightning_logs"), callbacks=[checkpoint], enable_progress_bar=False,
        limit_train_batches=args.limit_train_batches,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    metrics = trainer.test(model, dataloaders=test_loader, ckpt_path="best")[0]
    result = {
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "model_type": args.model_type,
        "protocol": "VOC2007 train frozen-backbone linear multi-label probe; val model selection; test evaluation",
        "best_validation_map": float(checkpoint.best_model_score),
        "test_map": float(metrics["test_map"]),
        "per_class_ap": {name: float(metrics[f"test_ap_{name}"]) for name in VOC_CLASSES},
    }
    result_path = output / "voc2007_multilabel.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    run_dir = os.environ.get("FMCA_HARNESS_RUN_DIR")
    if run_dir:
        with (Path(run_dir) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "stage": "voc2007_multilabel", **result}, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
