"""Lightning module for the CIFAR structure gate (seven matched variants).

Every variant consumes the SAME nested view-tree batch (same parents, same
encoded views, same backbone/projector budget); only the loss assembly
differs.  Variant names follow the frozen gate design:

  final_2view       1. flat FMCA on the leaf level, two views
  final_mview       2. flat FMCA on the leaf level, M views
  additive_2view    3. per-edge scores summed, two views per edge
  additive_mview    4. per-edge scores summed, M views (HAI/HFMCA family)
  amdim_cross       5. selected cross-scale pair scores summed
  product_only      6. ordered operator-product score, no endpoint closure
  product_endpoint  7. product + endpoint closure (full frozen objective)

Certificates reported from training batches are train-protocol diagnostics
only; the paper-grade certificate always comes from the frozen Stage-B/C
protocol on held-out data.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightning as L
import torch
from torch import Tensor, nn

from ..models import MLP
from .estimation import ChainFeatureBatch
from .objective import (
    batch_level_statistics,
    certificate_training_loss,
    cross_pair_score,
    identity_penalty,
    normalized_score,
    train_edge_operators,
    train_endpoint_operator,
)
from .stage_backbone import StageTappedCIFARResNet
from .triplet import compose_edge_operators

GATE_VARIANTS = (
    "final_2view",
    "final_mview",
    "additive_2view",
    "additive_mview",
    "amdim_cross",
    "product_only",
    "product_endpoint",
)


class HierarchyCertificateModule(L.LightningModule):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters({"config": config})
        model = config["model"]
        self.backbone = StageTappedCIFARResNet(width=int(model.get("backbone_width", 64)))
        self.level_stages: List[int] = [int(stage) for stage in model["level_stages"]]
        if any(late <= early for early, late in zip(self.level_stages, self.level_stages[1:])):
            raise ValueError(
                "level_stages must be strictly increasing: each edge advances one view "
                "refinement AND one backbone stage"
            )
        feature_dim = model.get("feature_dim", 128)
        dims = (
            [int(feature_dim)] * len(self.level_stages)
            if isinstance(feature_dim, int)
            else [int(dim) for dim in feature_dim]
        )
        if len(dims) != len(self.level_stages):
            raise ValueError("feature_dim must be scalar or one entry per level")
        hidden = model.get("head_hidden_dims", [512])
        activation = str(model.get("activation", "gelu"))
        # One projector per LEVEL, shared by every edge touching the level.
        self.projectors = nn.ModuleList(
            [
                MLP(self.backbone.stage_dims[stage], dims[level], hidden, activation)
                for level, stage in enumerate(self.level_stages)
            ]
        )
        self.variant = str(config.get("variant", "product_endpoint"))
        if self.variant not in GATE_VARIANTS:
            raise ValueError(f"variant must be one of {GATE_VARIANTS}")
        loss = config.get("loss", {})
        self.alpha = float(loss.get("alpha", 0.0))
        self.beta = float(loss.get("beta", 1.0))
        self.gamma = float(loss.get("gamma", 1.0))
        self.epsilon = float(loss.get("epsilon", 1e-6))
        self.closure_stop_grad = bool(loss.get("closure_stop_grad", False))
        pairs = config.get("cross_pairs", None)
        self.cross_pairs: Optional[List[Tuple[int, int]]] = (
            [(int(a), int(b)) for a, b in pairs] if pairs is not None else None
        )

    @property
    def config(self) -> Dict[str, Any]:
        return self.hparams["config"]

    @property
    def num_levels(self) -> int:
        return len(self.level_stages)

    def encode_level(self, images: Tensor, level: int) -> Tensor:
        """Project level-l views through the backbone up to the level's stage."""

        flat = images.flatten(0, -4)
        pooled = self.backbone.forward_stages(flat, up_to=self.level_stages[level])[-1]
        features = self.projectors[level](pooled)
        return features.reshape(*images.shape[:-3], -1)

    def feature_batch(self, batch: Dict[str, Any]) -> ChainFeatureBatch:
        chain_images: Sequence[Tensor] = batch["chain"]
        children_images: Sequence[Tensor] = batch["children"]
        if len(chain_images) != self.num_levels or len(children_images) != self.num_levels - 1:
            raise ValueError("batch levels do not match the configured hierarchy")
        chain = [self.encode_level(images, level) for level, images in enumerate(chain_images)]
        children = [
            self.encode_level(images, edge + 1) for edge, images in enumerate(children_images)
        ]
        endpoint = batch.get("endpoint")
        endpoint_features = (
            self.encode_level(endpoint, self.num_levels - 1) if endpoint is not None else None
        )
        return ChainFeatureBatch(
            chain=chain, children=children, endpoint_descendants=endpoint_features
        )

    @staticmethod
    def _truncate_views(features: ChainFeatureBatch, views: int) -> ChainFeatureBatch:
        return ChainFeatureBatch(
            chain=features.chain,
            children=[descendants[:, :views] for descendants in features.children],
            endpoint_descendants=features.endpoint_descendants,
        )

    def _whitening_penalty(self, moments: List[Tensor], levels: Sequence[int]) -> Tensor:
        return torch.stack([identity_penalty(moments[level]) for level in levels]).sum()

    def _flat_leaf_loss(self, features: ChainFeatureBatch, views: int) -> Tuple[Tensor, Dict[str, float]]:
        """Flat FMCA row: independent full-path descendants of the root are
        the star p(Y|X0) views of the classical flat method.  The parent
        feature is the conditional mean of the FIRST half of the views and
        the g side is the disjoint second half, so the cross moment carries
        no shared view noise (no I/M identity floor)."""

        if features.endpoint_descendants is None:
            raise ValueError(
                "flat variants require endpoint descendants: independent full-path "
                "views of the root (configure endpoint_descendants >= 2)"
            )
        leaf = features.endpoint_descendants[:, :views]
        if leaf.shape[1] < 2:
            raise ValueError("flat variants need at least two independent root views")
        half = leaf.shape[1] // 2
        mean = leaf.flatten(0, 1).mean(dim=0, keepdim=True)
        centered = leaf - mean.unsqueeze(0)
        f_side = centered[:, :half].mean(dim=1)
        g_side = centered[:, half:]
        cross = f_side.transpose(0, 1) @ g_side.mean(dim=1) / f_side.shape[0]
        score = normalized_score(cross)
        moment_f = f_side.transpose(0, 1) @ f_side / f_side.shape[0]
        pooled_g = g_side.flatten(0, 1)
        moment_g = pooled_g.transpose(0, 1) @ pooled_g / pooled_g.shape[0]
        whitening = identity_penalty(moment_f) + identity_penalty(moment_g)
        total = -score + self.gamma * whitening
        return total, {"leaf_score": float(score.detach()), "whitening": float(whitening.detach())}

    def _variant_loss(self, features: ChainFeatureBatch) -> Tuple[Tensor, Dict[str, float]]:
        if self.variant == "final_2view":
            return self._flat_leaf_loss(features, views=2)
        if self.variant == "final_mview":
            if features.endpoint_descendants is None:
                raise ValueError("final_mview requires endpoint descendants")
            return self._flat_leaf_loss(features, views=features.endpoint_descendants.shape[1])
        if self.variant == "additive_2view":
            features = self._truncate_views(features, views=2)
        if self.variant in {"additive_2view", "additive_mview"}:
            means, moments = batch_level_statistics(features)
            edges = train_edge_operators(features, means)
            score = torch.stack([normalized_score(edge) for edge in edges]).sum()
            whitening = self._whitening_penalty(moments, range(self.num_levels))
            total = -score + self.gamma * whitening
            return total, {"edge_score_sum": float(score.detach()), "whitening": float(whitening.detach())}
        if self.variant == "amdim_cross":
            means, moments = batch_level_statistics(features)
            pairs = self.cross_pairs or [
                (i, j) for i in range(self.num_levels) for j in range(1, self.num_levels) if i < j
            ]
            scores = [cross_pair_score(features, means, i, j) for i, j in pairs]
            score = torch.stack(scores).sum()
            whitening = self._whitening_penalty(moments, range(self.num_levels))
            total = -score + self.gamma * whitening
            return total, {"cross_score_sum": float(score.detach()), "whitening": float(whitening.detach())}
        if self.variant == "product_only":
            means, moments = batch_level_statistics(features)
            edges = train_edge_operators(features, means)
            score = normalized_score(compose_edge_operators(edges))
            whitening = self._whitening_penalty(moments, range(self.num_levels))
            total = -score + self.gamma * whitening
            return total, {"product_score": float(score.detach()), "whitening": float(whitening.detach())}
        terms = certificate_training_loss(
            features,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            epsilon=self.epsilon,
            closure_stop_grad=self.closure_stop_grad,
        )
        return terms.total, terms.as_metrics()

    def _shared_step(self, batch: Dict[str, Any], split: str) -> Tensor:
        features = self.feature_batch(batch)
        total, metrics = self._variant_loss(features)
        self.log(f"{split}/loss", total, on_step=False, on_epoch=True, prog_bar=True)
        for name, value in metrics.items():
            self.log(f"{split}/{name}", value, on_step=False, on_epoch=True)
        if split == "val":
            with torch.no_grad():
                means, _ = batch_level_statistics(features)
                edges = train_edge_operators(features, means)
                c_dir = train_endpoint_operator(features, means)
                c_comp = compose_edge_operators(edges)
                # Train-protocol diagnostic only; not the Stage-B/C certificate.
                self.log("val/closure_defect_frobenius", torch.linalg.matrix_norm(c_dir - c_comp))
                self.log("val_score", -total)
        return total

    def training_step(self, batch: Dict[str, Any], batch_index: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_index: int) -> None:
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
            total_epochs = int(config.get("scheduler_t_max", self.config["trainer"]["max_epochs"]))
            warmup_epochs = min(int(config.get("warmup_epochs", 10)), max(total_epochs - 1, 1))
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_epochs - warmup_epochs, 1)
            )
            if warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=warmup_epochs
                )
                scheduler: torch.optim.lr_scheduler.LRScheduler = (
                    torch.optim.lr_scheduler.SequentialLR(
                        optimizer, [warmup, cosine], milestones=[warmup_epochs]
                    )
                )
            else:
                scheduler = cosine
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
