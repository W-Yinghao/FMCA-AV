"""Stage-B shared per-level coordinates for the path-supported certificate.

Each level of the depth-aligned view hierarchy owns exactly ONE coordinate
system: mean, centered second moment, and symmetric inverse-square-root
whitener, fitted once on a held-out calibration split and then frozen.  The
same coordinates serve both the incoming and the outgoing edge of the level.
Fitting separate coordinates per edge is the forbidden design that the
control battery must expose (see controls.edgewise_calibrated_edges).

Centering precedes whitening.  Skipping the centering step keeps the
constant mode, which manufactures a trivial sigma = 1 direction on every
edge; ``centered=False`` exists only so the control battery can demonstrate
that failure.
"""

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import Tensor

from ..operators import inverse_sqrt_covariance


@dataclass
class LevelCoordinates:
    """Frozen affine coordinates z_tilde = (z - mean) @ whitener for one level."""

    mean: Tensor
    whitener: Tensor
    covariance: Tensor
    count: int
    ridge: float
    centered: bool

    @property
    def dim(self) -> int:
        return int(self.mean.shape[-1])

    def encode(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.dim:
            raise ValueError(
                f"features have dimension {features.shape[-1]}, coordinates expect {self.dim}"
            )
        if not features.is_floating_point():
            raise TypeError("features must have a floating-point dtype")
        if not bool(torch.isfinite(features.detach()).all()):
            raise ValueError("features contain non-finite values")
        # Promote the FEATURES up to the frozen-coordinate precision; casting
        # the frozen mean/whitener down to fp16 would silently corrupt every
        # downstream operator.
        target = torch.promote_types(features.dtype, self.whitener.dtype)
        if target in {torch.float16, torch.bfloat16}:
            target = torch.float32
        mean = self.mean.to(device=features.device, dtype=target)
        whitener = self.whitener.to(device=features.device, dtype=target)
        return (features.to(dtype=target) - mean) @ whitener

    def as_state(self) -> dict:
        return {
            "mean": self.mean.detach().cpu(),
            "whitener": self.whitener.detach().cpu(),
            "covariance": self.covariance.detach().cpu(),
            "count": self.count,
            "ridge": self.ridge,
            "centered": self.centered,
        }


def _flatten_level_features(features: Tensor) -> Tensor:
    if features.ndim == 2:
        return features
    if features.ndim == 3:
        return features.flatten(0, 1)
    raise ValueError(
        f"level features must be [samples, dim] or [parents, views, dim], got {tuple(features.shape)}"
    )


def fit_level_coordinates(
    features: Tensor,
    ridge: float = 1e-3,
    centered: bool = True,
) -> LevelCoordinates:
    """Fit one level's frozen coordinates on calibration features."""

    flat = _flatten_level_features(features)
    if flat.shape[0] < 2:
        raise ValueError("at least two samples are required to fit level coordinates")
    if not flat.is_floating_point():
        raise TypeError("level features must have a floating-point dtype")
    if not bool(torch.isfinite(flat.detach()).all()):
        raise ValueError("level features contain non-finite values")
    flat = flat.detach()
    if flat.dtype in {torch.float16, torch.bfloat16}:
        flat = flat.to(torch.float32)
    mean = flat.mean(dim=0, keepdim=True)
    work = flat - mean if centered else flat
    covariance = work.transpose(0, 1) @ work / flat.shape[0]
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    whitener = inverse_sqrt_covariance(covariance, ridge)
    if not centered:
        mean = torch.zeros_like(mean)
    return LevelCoordinates(
        mean=mean,
        whitener=whitener,
        covariance=covariance,
        count=int(flat.shape[0]),
        ridge=float(ridge),
        centered=bool(centered),
    )


def fit_chain_coordinates(
    level_features: Sequence[Tensor],
    ridge: float = 1e-3,
    centered: bool = True,
) -> List[LevelCoordinates]:
    """Fit one shared coordinate system per level of the chain."""

    if len(level_features) < 2:
        raise ValueError("a chain needs at least two levels")
    return [fit_level_coordinates(features, ridge=ridge, centered=centered) for features in level_features]
