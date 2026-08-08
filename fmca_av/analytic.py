"""Analytic reference spectra used by the E0/E1 gates."""

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor


@dataclass
class FiniteSpectrum:
    singular_values_with_constant: Tensor
    singular_values: Tensor
    eigenvalues: Tensor
    p_x: Tensor
    p_y: Tensor

    def as_metrics(self) -> Dict[str, object]:
        return {
            "singular_values_with_constant": self.singular_values_with_constant.tolist(),
            "singular_values": self.singular_values.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "constant_mode_removed": True,
            "trace_score": float(self.eigenvalues.sum()),
        }


def finite_channel_spectrum(joint: Tensor, tolerance: float = 1e-12) -> FiniteSpectrum:
    """Exact finite-alphabet spectrum of a joint probability table.

    The first singular value of ``diag(p_x)^-1/2 P diag(p_y)^-1/2`` is the
    constant mode and is removed explicitly.
    """

    if joint.ndim != 2:
        raise ValueError("joint probability table must be a matrix")
    joint = joint.to(dtype=torch.float64)
    if torch.any(joint < 0):
        raise ValueError("joint probabilities cannot be negative")
    total = joint.sum()
    if not torch.isfinite(total) or float(total) <= 0:
        raise ValueError("joint probability table must have positive finite mass")
    joint = joint / total
    p_x = joint.sum(dim=1)
    p_y = joint.sum(dim=0)
    if torch.any(p_x <= tolerance) or torch.any(p_y <= tolerance):
        raise ValueError("all supplied alphabet symbols must have positive marginal probability")
    normalized = p_x.rsqrt().unsqueeze(1) * joint * p_y.rsqrt().unsqueeze(0)
    all_singular = torch.linalg.svdvals(normalized)
    if not torch.isclose(all_singular[0], torch.ones_like(all_singular[0]), atol=1e-9, rtol=1e-9):
        raise RuntimeError("finite-channel constant singular mode is not numerically one")
    singular_values = all_singular[1:]
    return FiniteSpectrum(all_singular, singular_values, singular_values.square(), p_x, p_y)

