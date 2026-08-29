"""Closure plug-in study: bolt the EMA-closure regularizer onto standard
SSL objectives (FastSSL Barlow Twins / FastSSL VICReg / FroSSL).

Per method there are two rows sharing one data pipeline (full-fidelity
tree, M endpoint views) and one base objective on the endpoint views:

  base row   : L_base + sg-backbone(alpha * edges + gamma)   -- the
               hierarchy heads train as pure MEASUREMENT heads on
               detached backbone features; the representation is shaped
               by the base objective alone.
  plugin row : L_base + alpha * edges + beta * closure(EMA) + gamma --
               the closure gradient flows into the backbone.

The only difference reaching the backbone is the closure regularizer,
so the pair isolates its effect on both the certificate and downstream
accuracy.  Certificates are measured post-hoc by the frozen Stage-B/C
protocol in both rows.
"""

from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from ..baselines import (
    FastSSLProjector,
    FroSSLProjector,
    fastssl_barlow_twins_loss,
    fastssl_vicreg_loss,
    frossl_loss,
)
from .estimation import ChainFeatureBatch
from .hierarchy_module import HierarchyCertificateModule
from .objective import (
    train_edge_operators,
    train_endpoint_operator,
    trace_score_edges,
    whiten_chain_batch,
)
from .triplet import compose_edge_operators

BASE_OBJECTIVES = ("barlow_twins", "vicreg", "frossl")


class PluginSSLModule(HierarchyCertificateModule):
    """Standard SSL objective with an optional EMA-closure plug-in."""

    def __init__(self, config: Dict[str, Any]) -> None:
        config = dict(config)
        config.setdefault("variant", "product_endpoint")
        super().__init__(config)
        base = config["base"]
        self.base_name = str(base["name"])
        if self.base_name not in BASE_OBJECTIVES:
            raise ValueError(f"base.name must be one of {BASE_OBJECTIVES}")
        representation = self.backbone.output_dim
        if self.base_name == "frossl":
            self.base_projector = FroSSLProjector(
                representation,
                int(base.get("projector_hidden", 2048)),
                int(base.get("projector_dim", 1024)),
            )
        else:
            # FastSSL's small projector takes (input_dim, projection_dim).
            self.base_projector = FastSSLProjector(
                representation, int(base.get("projector_dim", 256))
            )
        self.base_parameters = {
            "off_diagonal_weight": float(base.get("off_diagonal_weight", 1.0 / 256.0)),
            "invariance_weight": float(base.get("invariance_weight", 25.0)),
            "variance_weight": float(base.get("variance_weight", 25.0)),
            "frossl_invariance": float(base.get("frossl_invariance", 2.0)),
        }
        self.plugin_enabled = bool(config.get("plugin", {}).get("enabled", False))

    def _base_loss(self, endpoint_images: Tensor) -> Tensor:
        batch, views = endpoint_images.shape[:2]
        representations = self.backbone(endpoint_images.flatten(0, 1))
        projections = self.base_projector(representations).reshape(batch, views, -1)
        if self.base_name == "barlow_twins":
            return fastssl_barlow_twins_loss(
                projections, self.base_parameters["off_diagonal_weight"]
            )
        if self.base_name == "vicreg":
            return fastssl_vicreg_loss(
                projections,
                self.base_parameters["invariance_weight"],
                self.base_parameters["variance_weight"],
            )
        return frossl_loss(projections, self.base_parameters["frossl_invariance"])

    def _hierarchy_terms(self, batch: Dict[str, Any], detach_backbone: bool) -> Tuple[Tensor, Dict[str, float]]:
        if detach_backbone:
            features_raw = self._feature_batch_detached(batch)
        else:
            features_raw = self.feature_batch(batch)
        edge_traces = trace_score_edges(features_raw)
        edge_sum = torch.stack(edge_traces).sum()
        whitened, moments = whiten_chain_batch(
            features_raw, ridge=self.ridge, detach_whitener=self.detach_whitener
        )
        edges = train_edge_operators(whitened)
        c_comp = compose_edge_operators(edges)
        c_dir = train_endpoint_operator(whitened)
        if self.ema_target_momentum > 0:
            with torch.no_grad():
                if float(self.ema_initialized) == 0.0:
                    self.ema_c_dir.copy_(c_dir)
                    self.ema_initialized.fill_(1.0)
                else:
                    self.ema_c_dir.mul_(self.ema_target_momentum).add_(
                        c_dir, alpha=1.0 - self.ema_target_momentum
                    )
            closure = (self.ema_c_dir - c_comp).square().sum() / (
                self.ema_c_dir.square().sum() + self.epsilon
            )
        else:
            closure = (c_dir.detach() - c_comp).square().sum() / (
                c_dir.detach().square().sum() + self.epsilon
            )
        whitening = self._whitening_penalty(moments, range(self.num_levels))
        terms = -self.alpha * edge_sum + self.gamma * whitening
        if self.plugin_enabled:
            terms = terms + self.beta * closure
        metrics = {
            "edge_trace_sum": float(edge_sum.detach()),
            "closure_ratio": float(closure.detach()),
            "whitening": float(whitening.detach()),
        }
        return terms, metrics

    def _feature_batch_detached(self, batch: Dict[str, Any]) -> ChainFeatureBatch:
        """Encode with backbone features detached: heads train, backbone doesn't."""

        def encode(images: Tensor, level: int) -> Tensor:
            flat = images.flatten(0, -4)
            with torch.no_grad():
                pooled = self.backbone.forward_stages(flat, up_to=self.level_stages[level])[-1]
            features = self.projectors[level](pooled)
            return features.reshape(*images.shape[:-3], -1)

        chain = [encode(images, level) for level, images in enumerate(batch["chain"])]
        children = [encode(images, edge + 1) for edge, images in enumerate(batch["children"])]
        endpoint = encode(batch["endpoint"], self.num_levels - 1)
        return ChainFeatureBatch(chain=chain, children=children, endpoint_descendants=endpoint)

    def _shared_step(self, batch: Dict[str, Any], split: str) -> Tensor:
        base_loss = self._base_loss(batch["endpoint"])
        hierarchy_terms, metrics = self._hierarchy_terms(
            batch, detach_backbone=not self.plugin_enabled
        )
        total = base_loss + hierarchy_terms
        self.log(f"{split}/loss", total, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{split}/base_loss", base_loss, on_step=False, on_epoch=True)
        for name, value in metrics.items():
            self.log(f"{split}/{name}", value, on_step=False, on_epoch=True)
        if split == "val":
            self.log("val_score", -total)
        return total
