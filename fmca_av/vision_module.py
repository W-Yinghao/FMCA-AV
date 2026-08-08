"""Lightning FMCA-AV module for multi-view image representation learning."""

from typing import Any, Dict, Optional, Tuple

import lightning as L
import torch
from torch import Tensor, nn

from .models import MLP, ParentAggregator
from .objectives import fmca_score
from .operators import estimate_moments
from .backbones import build_backbone


class VisionFMCAAV(L.LightningModule):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters({"config": config})
        model = config["model"]
        self.backbone = build_backbone(
            str(model.get("backbone", "resnet18_cifar")),
            width=int(model.get("backbone_width", 64)),
        )
        representation_dim = self.backbone.output_dim
        feature_dim = int(model["feature_dim"])
        self.parent_feature_source = str(model.get("parent_feature_source", "backbone"))
        if self.parent_feature_source not in {"backbone", "g"}:
            raise ValueError("model.parent_feature_source must be backbone or g")
        aggregator_dim = feature_dim if self.parent_feature_source == "g" else representation_dim
        self.parent_aggregator = ParentAggregator(
            aggregator_dim,
            mode=str(model.get("parent_aggregation", "mean")),
            hidden_dim=int(model.get("aggregator_hidden_dim", aggregator_dim)),
            num_views=int(config["data"]["num_views"]),
        )
        head_hidden = model.get("head_hidden_dims", [512])
        f_head_hidden = model.get("f_head_hidden_dims", head_hidden)
        g_head_hidden = model.get("g_head_hidden_dims", head_hidden)
        activation = str(model.get("activation", "gelu"))
        self.shared_head = bool(model.get("shared_head", False))
        self.stop_gradient = str(model.get("stop_gradient", "none"))
        if self.stop_gradient not in {"none", "f", "g"}:
            raise ValueError("model.stop_gradient must be none, f, or g")
        if self.shared_head:
            if self.parent_aggregator.output_dim != representation_dim:
                raise ValueError("shared_head requires the parent aggregator output dimension to match the backbone representation")
            if list(f_head_hidden) != list(g_head_hidden):
                raise ValueError("shared_head requires identical f/g hidden dimensions")
            shared = MLP(representation_dim, feature_dim, f_head_hidden, activation)
            self.f_head = shared
            self.g_head = shared
        else:
            self.f_head = MLP(self.parent_aggregator.output_dim, feature_dim, f_head_hidden, activation)
            self.g_head = MLP(representation_dim, feature_dim, g_head_hidden, activation)

    @property
    def config(self) -> Dict[str, Any]:
        return self.hparams["config"]

    def feature_maps(
        self, views: Tensor, parent_view: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if views.ndim != 5:
            raise ValueError("views must have shape [batch, views, channels, height, width]")
        batch, count = views.shape[:2]
        representations = self.backbone(views.flatten(0, 1)).reshape(batch, count, -1)
        g_features = self.g_head(representations.flatten(0, 1)).reshape(batch, count, -1)
        if parent_view is not None:
            if self.parent_aggregator.mode != "raw":
                raise ValueError("an explicit raw parent requires parent_aggregation=raw")
            parent_representation = self.backbone(parent_view)
            parent_features = (
                self.g_head(parent_representation)
                if self.parent_feature_source == "g" else parent_representation
            )
            f_features = self.f_head(parent_features)
        else:
            parent_features = g_features if self.parent_feature_source == "g" else representations
            f_features = self.f_head(self.parent_aggregator(parent_features))
        return f_features, g_features, representations

    def _shared_step(self, batch: Tuple[Tensor, Tensor, Tensor], split: str) -> Tensor:
        views = batch[0]
        parent_view = batch[3] if len(batch) > 3 else None
        f_features, g_features, _ = self.feature_maps(views, parent_view)
        if self.stop_gradient == "f":
            f_features = f_features.detach()
        elif self.stop_gradient == "g":
            g_features = g_features.detach()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            from torch.distributed.nn.functional import all_gather
            f_features = torch.cat(tuple(all_gather(f_features.contiguous())), dim=0)
            g_features = torch.cat(tuple(all_gather(g_features.contiguous())), dim=0)
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
        batch_size, views_per_parent = views.shape[:2]
        world_size = torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1
        self.log(f"{split}/loss", loss, on_step=False, on_epoch=True, prog_bar=split == "val")
        self.log(f"{split}/score", score, on_step=False, on_epoch=True, prog_bar=True)
        encoded_per_parent = views_per_parent + (1 if parent_view is not None else 0)
        self.log(f"{split}/encoded_views", float(batch_size * encoded_per_parent * world_size), on_step=False, on_epoch=True, reduce_fx="sum", sync_dist=True)
        if split == "val":
            self.log("val_score", score, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def configure_optimizers(self) -> object:
        config = self.config["optimizer"]
        name = str(config.get("name", "adamw"))
        if name == "adamw":
            optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=float(config["learning_rate"]),
                weight_decay=float(config.get("weight_decay", 0.0)),
            )
        elif name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=float(config["learning_rate"]),
                momentum=float(config.get("momentum", 0.9)),
                weight_decay=float(config.get("weight_decay", 0.0)),
            )
        else:
            raise ValueError("optimizer.name must be adamw or sgd")
        if str(config.get("scheduler", "none")) == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(config.get("scheduler_t_max", self.config["trainer"]["max_epochs"])),
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
