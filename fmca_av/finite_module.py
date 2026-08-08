"""Lightning implementation of FMCA-AV on finite alphabets."""

from typing import Any, Dict, Tuple

import lightning as L
import torch
from torch import Tensor, nn

from .data.finite import normalized_joint, sample_finite_conditionals
from .objectives import fmca_score
from .operators import estimate_moments


class FiniteFMCAAV(L.LightningModule):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters({"config": config})
        joint = normalized_joint(config["data"])
        feature_dim = int(config["model"]["feature_dim"])
        self.f = nn.Embedding(joint.shape[0], feature_dim)
        self.g = nn.Embedding(joint.shape[1], feature_dim)
        conditional = (joint / joint.sum(dim=1, keepdim=True)).float()
        self.register_buffer("conditional", conditional)

    @property
    def config(self) -> Dict[str, Any]:
        return self.hparams["config"]

    def feature_maps(self, x: Tensor, y_views: Tensor) -> Tuple[Tensor, Tensor]:
        return self.f(x), self.g(y_views)

    def _shared_step(self, batch: Tuple[Tensor], split: str) -> Tensor:
        x = batch[0]
        y = sample_finite_conditionals(x, self.conditional, int(self.config["data"]["num_views"]))
        f_features, g_features = self.feature_maps(x, y)
        moments = estimate_moments(f_features, g_features, centered=True)
        objective = self.config["objective"]
        score = fmca_score(
            moments,
            str(objective["name"]),
            ridge=float(objective.get("ridge", 1e-3)),
            logdet_margin=float(objective.get("logdet_margin", 1e-6)),
        )
        loss = -score
        self.log(f"{split}/loss", loss, on_step=False, on_epoch=True)
        self.log(f"{split}/score", score, on_step=False, on_epoch=True, prog_bar=True)
        if split == "val":
            self.log("val_score", score, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: Tuple[Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.config["optimizer"]
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(optimizer["learning_rate"]),
            weight_decay=float(optimizer.get("weight_decay", 0.0)),
        )

