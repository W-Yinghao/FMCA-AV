"""Lightning training modules for FMCA-AV."""

from typing import Any, Dict, Tuple

import lightning as L
import torch
from torch import Tensor

from .data.gaussian import sample_conditionals
from .models import MLP
from .objectives import fmca_score
from .operators import estimate_moments


class GaussianFMCAAV(L.LightningModule):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters({"config": config})
        model = config["model"]
        input_dim = int(config["data"].get("dimension", 1))
        hidden_dims = model.get("hidden_dims", [128, 128])
        self.f = MLP(input_dim, int(model["feature_dim"]), hidden_dims, str(model.get("activation", "gelu")))
        self.g = MLP(input_dim, int(model["feature_dim"]), hidden_dims, str(model.get("activation", "gelu")))

    @property
    def config(self) -> Dict[str, Any]:
        return self.hparams["config"]

    def feature_maps(self, x: Tensor, y_views: Tensor) -> Tuple[Tensor, Tensor]:
        shape = y_views.shape
        f_features = self.f(x)
        g_features = self.g(y_views.reshape(-1, shape[-1])).reshape(shape[0], shape[1], -1)
        return f_features, g_features

    def _shared_step(self, batch: Tuple[Tensor], split: str) -> Tensor:
        x = batch[0]
        data = self.config["data"]
        y_views = sample_conditionals(x, int(data["num_views"]), float(data["noise_variance"]))
        f_features, g_features = self.feature_maps(x, y_views)
        moments = estimate_moments(
            f_features,
            g_features,
            centered=bool(self.config["model"].get("centered", True)),
        )
        objective = self.config["objective"]
        score = fmca_score(
            moments,
            str(objective["name"]),
            ridge=float(objective.get("ridge", 1e-3)),
            logdet_margin=float(objective.get("logdet_margin", 1e-6)),
        )
        loss = -score
        self.log(f"{split}/loss", loss, prog_bar=split == "val", on_step=False, on_epoch=True)
        self.log(f"{split}/score", score, prog_bar=True, on_step=False, on_epoch=True)
        if split == "val":
            self.log("val_score", score, prog_bar=False, on_step=False, on_epoch=True)
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
