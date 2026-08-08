"""Frozen-backbone Lightning linear evaluation protocol."""

from typing import Any, Dict, Tuple

import lightning as L
import torch
from torch import Tensor, nn


class LinearProbe(L.LightningModule):
    def __init__(self, backbone: nn.Module, representation_dim: int, classes: int, config: Dict[str, Any]) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.classifier = nn.Linear(representation_dim, classes)
        self.probe_config = config
        self.save_hyperparameters(ignore=["backbone"])

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def forward(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.backbone(images)
        return self.classifier(features)

    def _shared_step(self, batch: Tuple[Tensor, Tensor], split: str) -> Tensor:
        images, labels = batch
        logits = self(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        top_count = min(5, logits.shape[1])
        top5_accuracy = (
            (logits.topk(top_count, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean()
        )
        self.log(f"{split}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{split}/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{split}/top5_accuracy", top5_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        if split == "val":
            self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> object:
        optimizer = torch.optim.SGD(
            self.classifier.parameters(),
            lr=float(self.probe_config.get("learning_rate", 0.1)),
            momentum=float(self.probe_config.get("momentum", 0.9)),
            weight_decay=float(self.probe_config.get("weight_decay", 0.0)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(self.probe_config.get("max_epochs", 100))
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class FineTuneClassifier(L.LightningModule):
    """End-to-end low-label evaluation with a trainable pretrained backbone."""

    def __init__(self, backbone: nn.Module, representation_dim: int, classes: int, config: Dict[str, Any]) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(True)
        self.classifier = nn.Linear(representation_dim, classes)
        self.finetune_config = config
        self.save_hyperparameters(ignore=["backbone"])

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.backbone(images))

    def _shared_step(self, batch: Tuple[Tensor, Tensor], split: str) -> Tensor:
        images, labels = batch
        logits = self(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        accuracy = (logits.argmax(1) == labels).float().mean()
        top_count = min(5, logits.shape[1])
        top5 = (logits.topk(top_count, 1).indices == labels[:, None]).any(1).float().mean()
        self.log(f"{split}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{split}/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{split}/top5_accuracy", top5, on_step=False, on_epoch=True, sync_dist=True)
        if split == "val":
            self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> object:
        learning_rate = float(self.finetune_config.get("finetune_learning_rate", 0.01))
        classifier_multiplier = float(self.finetune_config.get("classifier_lr_multiplier", 1.0))
        optimizer = torch.optim.SGD(
            [
                {"params": self.backbone.parameters(), "lr": learning_rate},
                {"params": self.classifier.parameters(), "lr": learning_rate * classifier_multiplier},
            ],
            momentum=float(self.finetune_config.get("momentum", 0.9)),
            weight_decay=float(self.finetune_config.get("finetune_weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(self.finetune_config.get("max_epochs", 100))
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
