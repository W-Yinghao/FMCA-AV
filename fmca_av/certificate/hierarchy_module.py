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
from ..objectives import trace_score
from ..operators import estimate_moments
from .estimation import ChainFeatureBatch
from .objective import (
    certificate_training_loss,
    cholesky_whitener,
    cross_pair_score,
    identity_penalty,
    normalized_score,
    train_edge_operators,
    train_endpoint_operator,
    whiten_chain_batch,
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
        self.backbone = StageTappedCIFARResNet(
            width=int(model.get("backbone_width", 64)), stem=str(model.get("stem", "cifar"))
        )
        # Locality test (E-A3): load a backbone another arm trained and freeze
        # it bit-for-bit, so only the projectors and operator heads can move.
        # If the defect still falls, it is a readout property, not a
        # representation property.
        self.frozen_backbone = bool(model.get("freeze_backbone", False))
        backbone_source = str(model.get("backbone_checkpoint", ""))
        if backbone_source:
            payload = torch.load(backbone_source, map_location="cpu", weights_only=False)
            state = {
                key[len("backbone."):]: value
                for key, value in payload["state_dict"].items()
                if key.startswith("backbone.")
            }
            if not state:
                raise ValueError(f"no backbone weights in {backbone_source}")
            self.backbone.load_state_dict(state)
        elif self.frozen_backbone:
            raise ValueError("freeze_backbone needs backbone_checkpoint: freezing a "
                             "random encoder measures nothing")
        if self.frozen_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
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
        self.ridge = float(loss.get("ridge", 0.1))
        whitening_mode = str(loss.get("whitening_mode", "detached"))
        if whitening_mode not in {"detached", "differentiable"}:
            raise ValueError("loss.whitening_mode must be detached or differentiable")
        self.detach_whitener = whitening_mode == "detached"
        self.closure_stop_grad = bool(loss.get("closure_stop_grad", False))
        pairs = config.get("cross_pairs", None)
        self.cross_pairs: Optional[List[Tuple[int, int]]] = (
            [(int(a), int(b)) for a, b in pairs] if pairs is not None else None
        )
        # Flat-row recipe: "split_half_whitened" (gate v3/v4) or
        # "faithful_trace" (gate v5+), which replicates the repo's formal
        # flat FMCA-AV estimator: f = f_head(mean of projected views),
        # differentiable-whitened trace score at ridge 1e-3 (the recipe
        # behind the 85.4/89.4% historical rows).
        self.flat_recipe = str(loss.get("flat_recipe", "split_half_whitened"))
        if self.flat_recipe not in {"split_half_whitened", "faithful_trace"}:
            raise ValueError("loss.flat_recipe must be split_half_whitened or faithful_trace")
        # Additive-family recipe: "whitened" scores in shared pooled
        # coordinates (v3-v6b) or "faithful_trace" per-operator FMCA scores
        # (v6c+).  The additive/AMDIM baselines never used shared-interface
        # whitening historically (per-stage losses whiten per stage), and a
        # pure edge-score objective under a shared differentiable whitener
        # farms the thin-subset estimation gap (guard trip at 5.03).  The
        # compositional rows keep shared coordinates: composition semantics
        # require one interior basis, and their loss has no per-edge score
        # incentive (alpha = 0).
        self.additive_recipe = str(loss.get("additive_recipe", "whitened"))
        if self.additive_recipe not in {"whitened", "faithful_trace"}:
            raise ValueError("loss.additive_recipe must be whitened or faithful_trace")
        # Product-row recipe: "whitened" (v6b) or "faithful_bootstrap"
        # (v6d+): faithful trace endpoint reward + small-alpha faithful
        # per-edge bootstrap (the frozen algebra's "alpha small" clause)
        # + the closure ratio in shared coordinates with beta rescaled to
        # compete at trace magnitude.  Cures the composition cold-start
        # (v6b: closure ratio pinned at 1, edges decaying to zero).
        self.product_recipe = str(loss.get("product_recipe", "whitened"))
        if self.product_recipe not in {"whitened", "faithful_bootstrap"}:
            raise ValueError("loss.product_recipe must be whitened or faithful_bootstrap")
        # alpha_schedule "cosine_to_zero" anneals the edge bootstrap away so
        # the converged objective is the frozen algebra's alpha -> 0 form.
        self.alpha_schedule = str(loss.get("alpha_schedule", "constant"))
        if self.alpha_schedule not in {"constant", "cosine_to_zero"}:
            raise ValueError("loss.alpha_schedule must be constant or cosine_to_zero")
        # Optional leaf-level flat-style reward inside the bootstrap recipe:
        # f_head(mean of endpoint views) vs views, all at the final stage --
        # the engine behind the 84% flat anchor, added on top of the
        # depth-aligned endpoint term ("flat objective + closure
        # regularizer" form).
        self.leaf_reward_weight = float(loss.get("leaf_reward_weight", 0.0))
        # Curriculum: train the leaf (flat) term alone for the first N
        # epochs, then switch on the full compositional objective
        # ("closure fine-tuning" of a flat-quality representation).
        self.curriculum_epochs = int(loss.get("curriculum_epochs", 0))
        if self.curriculum_epochs > 0 and self.leaf_reward_weight <= 0:
            raise ValueError("curriculum_epochs requires leaf_reward_weight > 0")
        # Operator-level EMA closure target: the closure ratio chases a
        # slowly-moving average of C_dir instead of the live batch operator
        # (the target-network fix for the raw stop-grad crash).
        self.ema_target_momentum = float(loss.get("ema_target_momentum", 0.0))
        self.flat_f_head: Optional[MLP] = None
        needs_flat_head = (
            self.variant in {"final_2view", "final_mview"} and self.flat_recipe == "faithful_trace"
        ) or (
            self.variant == "product_endpoint"
            and self.product_recipe == "faithful_bootstrap"
            and self.leaf_reward_weight > 0
        )
        if needs_flat_head:
            self.flat_f_head = MLP(dims[-1], dims[-1], hidden, activation)
        if self.ema_target_momentum > 0:
            self.register_buffer("ema_c_dir", torch.zeros(dims[0], dims[-1]))
            self.register_buffer("ema_initialized", torch.zeros(1))

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode so BatchNorm cannot drift."""

        super().train(mode)
        if getattr(self, "frozen_backbone", False):
            self.backbone.eval()
        return self

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
        the star p(Y|X0) views of the classical flat method.

        faithful_trace (gate v5+): the repo's formal flat FMCA-AV estimator
        (f = f_head(mean of projected views), whitened trace at ridge 1e-3).
        split_half_whitened (v3/v4): parent = conditional mean of the first
        half of the views, g = the disjoint second half."""

        if features.endpoint_descendants is None:
            raise ValueError(
                "flat variants require endpoint descendants: independent full-path "
                "views of the root (configure endpoint_descendants >= 2)"
            )
        leaf = features.endpoint_descendants[:, :views]
        if leaf.shape[1] < 2:
            raise ValueError("flat variants need at least two independent root views")
        if self.flat_recipe == "faithful_trace":
            assert self.flat_f_head is not None
            f_features = self.flat_f_head(leaf.mean(dim=1))
            moments = estimate_moments(f_features, leaf, centered=True)
            score = trace_score(moments, ridge=1e-3)
            total = -score
            return total, {"flat_trace_score": float(score.detach())}
        half = leaf.shape[1] // 2
        mean = leaf.flatten(0, 1).mean(dim=0, keepdim=True)
        centered = leaf - mean.unsqueeze(0)
        pooled = centered.flatten(0, 1)
        moment = pooled.transpose(0, 1) @ pooled / pooled.shape[0]
        # Leaf-level terms only: the flat control must not train the unused
        # non-leaf projectors through shared penalties.
        white = centered @ cholesky_whitener(moment.detach(), self.ridge)
        f_side = white[:, :half].mean(dim=1)
        g_side = white[:, half:]
        cross = f_side.transpose(0, 1) @ g_side.mean(dim=1) / f_side.shape[0]
        score = normalized_score(cross)
        whitening = identity_penalty(moment)
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
            if self.additive_recipe == "faithful_trace":
                scores = [
                    trace_score(
                        estimate_moments(features.chain[edge], features.children[edge], centered=True),
                        ridge=1e-3,
                    )
                    for edge in range(self.num_levels - 1)
                ]
                score = torch.stack(scores).sum()
                return -score, {"edge_trace_sum": float(score.detach())}
            whitened, moments = whiten_chain_batch(features, ridge=self.ridge, detach_whitener=self.detach_whitener)
            edges = train_edge_operators(whitened)
            score = torch.stack([normalized_score(edge) for edge in edges]).sum()
            whitening = self._whitening_penalty(moments, range(self.num_levels))
            total = -score + self.gamma * whitening
            return total, {"edge_score_sum": float(score.detach()), "whitening": float(whitening.detach())}
        if self.variant == "amdim_cross":
            pairs = self.cross_pairs or [
                (i, j) for i in range(self.num_levels) for j in range(1, self.num_levels) if i < j
            ]
            if self.additive_recipe == "faithful_trace":
                scores = [
                    trace_score(
                        estimate_moments(features.chain[i], features.children[j - 1], centered=True),
                        ridge=1e-3,
                    )
                    for i, j in pairs
                ]
                score = torch.stack(scores).sum()
                return -score, {"cross_trace_sum": float(score.detach())}
            whitened, moments = whiten_chain_batch(features, ridge=self.ridge, detach_whitener=self.detach_whitener)
            scores = [cross_pair_score(whitened, None, i, j) for i, j in pairs]
            score = torch.stack(scores).sum()
            whitening = self._whitening_penalty(moments, range(self.num_levels))
            total = -score + self.gamma * whitening
            return total, {"cross_score_sum": float(score.detach()), "whitening": float(whitening.detach())}
        if self.variant == "product_endpoint" and self.product_recipe == "faithful_bootstrap":
            if features.endpoint_descendants is None:
                raise ValueError("product_endpoint requires endpoint descendants")
            alpha = self.alpha
            if self.alpha_schedule == "cosine_to_zero":
                try:
                    progress = min(self.current_epoch / max(self.trainer.max_epochs, 1), 1.0)
                except RuntimeError:
                    progress = 0.0
                import math

                alpha = self.alpha * 0.5 * (1.0 + math.cos(math.pi * progress))
            reward_dir = trace_score(
                estimate_moments(features.chain[0], features.endpoint_descendants, centered=True),
                ridge=1e-3,
            )
            edge_traces = [
                trace_score(
                    estimate_moments(features.chain[edge], features.children[edge], centered=True),
                    ridge=1e-3,
                )
                for edge in range(self.num_levels - 1)
            ]
            edge_sum = torch.stack(edge_traces).sum()
            whitened, moments = whiten_chain_batch(
                features, ridge=self.ridge, detach_whitener=self.detach_whitener
            )
            shared_edges = train_edge_operators(whitened)
            c_comp = compose_edge_operators(shared_edges)
            c_dir = train_endpoint_operator(whitened)
            closure_target = c_dir.detach() if self.closure_stop_grad else c_dir
            closure_denominator = (
                c_dir.detach().square().sum() if self.closure_stop_grad else c_dir.square().sum()
            )
            closure = (closure_target - c_comp).square().sum() / (closure_denominator + self.epsilon)
            whitening = self._whitening_penalty(moments, range(self.num_levels))
            leaf_reward = None
            if self.leaf_reward_weight > 0:
                assert self.flat_f_head is not None
                leaf_views = features.endpoint_descendants
                f_features = self.flat_f_head(leaf_views.mean(dim=1))
                leaf_reward = trace_score(
                    estimate_moments(f_features, leaf_views, centered=True), ridge=1e-3
                )
            if self.ema_target_momentum > 0:
                with torch.no_grad():
                    momentum = self.ema_target_momentum
                    if float(self.ema_initialized) == 0.0:
                        self.ema_c_dir.copy_(c_dir)
                        self.ema_initialized.fill_(1.0)
                    else:
                        self.ema_c_dir.mul_(momentum).add_(c_dir, alpha=1.0 - momentum)
                closure = (self.ema_c_dir - c_comp).square().sum() / (
                    self.ema_c_dir.square().sum() + self.epsilon
                )
            in_warmup_phase = False
            try:
                in_warmup_phase = (
                    self.curriculum_epochs > 0 and self.current_epoch < self.curriculum_epochs
                )
            except RuntimeError:
                pass
            if in_warmup_phase:
                total = -self.leaf_reward_weight * leaf_reward + self.gamma * whitening
            else:
                total = (
                    -reward_dir
                    - alpha * edge_sum
                    + self.beta * closure
                    + self.gamma * whitening
                )
                if leaf_reward is not None:
                    total = total - self.leaf_reward_weight * leaf_reward
            metrics = {
                "dir_trace": float(reward_dir.detach()),
                "edge_trace_sum": float(edge_sum.detach()),
                "closure_ratio": float(closure.detach()),
                "whitening": float(whitening.detach()),
                "alpha_effective": float(alpha),
            }
            if leaf_reward is not None:
                metrics["leaf_trace"] = float(leaf_reward.detach())
            return total, metrics
        if self.variant == "product_only":
            whitened, moments = whiten_chain_batch(features, ridge=self.ridge, detach_whitener=self.detach_whitener)
            edges = train_edge_operators(whitened)
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
            ridge=self.ridge,
            detach_whitener=self.detach_whitener,
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
                whitened, _ = whiten_chain_batch(features, ridge=self.ridge, detach_whitener=True)
                edges = train_edge_operators(whitened)
                c_dir = train_endpoint_operator(whitened)
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
        trainable = [p for p in self.parameters() if p.requires_grad]
        config = self.config["optimizer"]
        name = str(config.get("name", "adamw"))
        if name == "adamw":
            optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                trainable,
                lr=float(config["learning_rate"]),
                weight_decay=float(config.get("weight_decay", 0.0)),
            )
        elif name == "sgd":
            optimizer = torch.optim.SGD(
                trainable,
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
