"""FMCA-AV moment operators and finite-dimensional spectral calibration.

The implementation follows equations (2)--(4) of the paper.  In particular,
``R_g`` is the mean of per-view outer products.  Only the cross moment uses the
conditional view mean.  Centering is enabled by default to remove the constant
mode before calibration, as required by the experiment plan.
"""

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor


@dataclass
class FMCAMoments:
    r_f: Tensor
    r_g: Tensor
    p_fg: Tensor
    count_x: int
    count_y: int
    centered: bool


@dataclass
class SpectralCalibration:
    mean_f: Tensor
    mean_g: Tensor
    transform_f: Tensor
    transform_g: Tensor
    singular_values: Tensor
    eigenvalues: Tensor
    ridge: float
    centered: bool

    def encode_f(self, features: Tensor) -> Tensor:
        return (features - self.mean_f) @ self.transform_f

    def encode_g(self, features: Tensor) -> Tensor:
        return (features - self.mean_g) @ self.transform_g

    def as_metrics(self) -> Dict[str, object]:
        eigenvalues = self.eigenvalues.detach().cpu()
        singular_values = self.singular_values.detach().cpu()
        return {
            "centered": self.centered,
            "ridge": self.ridge,
            "ridge_mode": "relative_to_mean_marginal_variance",
            "singular_values": singular_values.tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "trace_score": float(eigenvalues.sum()),
            "logdet_score": float((-torch.log1p(-eigenvalues.clamp(max=1.0 - 1e-7))).sum()),
        }


def _check_features(f: Tensor, g_views: Tensor) -> None:
    if f.ndim != 2:
        raise ValueError(f"f must have shape [parents, features], got {tuple(f.shape)}")
    if g_views.ndim != 3:
        raise ValueError(
            f"g_views must have shape [parents, views, features], got {tuple(g_views.shape)}"
        )
    if f.shape[0] != g_views.shape[0]:
        raise ValueError("f and g_views must have the same number of parent samples")
    if f.shape[1] != g_views.shape[2]:
        raise ValueError("f and g must have the same feature dimension")
    if f.shape[0] < 2 or g_views.shape[1] < 1:
        raise ValueError("at least two parents and one conditional view are required")


def estimate_moments(f: Tensor, g_views: Tensor, centered: bool = True) -> FMCAMoments:
    """Estimate the three matrices in the FMCA-AV block operator.

    ``f`` is ``[B, K]`` and ``g_views`` is ``[B, M, K]``.  The estimators are
    normalized population moments (division by B and B*M), matching the
    objective rather than an unbiased sample-covariance convention.
    """

    _check_features(f, g_views)
    batch, views, _ = g_views.shape
    if centered:
        f_work = f - f.mean(dim=0, keepdim=True)
        g_work = g_views - g_views.mean(dim=(0, 1), keepdim=True)
    else:
        f_work = f
        g_work = g_views

    g_mean = g_work.mean(dim=1)
    r_f = f_work.transpose(0, 1) @ f_work / batch
    r_g = torch.einsum("bmk,bml->kl", g_work, g_work) / (batch * views)
    p_fg = f_work.transpose(0, 1) @ g_mean / batch
    r_f = 0.5 * (r_f + r_f.transpose(0, 1))
    r_g = 0.5 * (r_g + r_g.transpose(0, 1))
    return FMCAMoments(r_f, r_g, p_fg, batch, batch * views, centered)


def _inverse_sqrt(matrix: Tensor, ridge: float) -> Tensor:
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    dimension = matrix.shape[0]
    scale = matrix.diagonal().abs().mean().detach().clamp_min(1.0)
    regularized = matrix + (ridge * scale) * torch.eye(
        dimension, dtype=matrix.dtype, device=matrix.device
    )
    values, vectors = torch.linalg.eigh(regularized)
    values = values.clamp_min(torch.finfo(matrix.dtype).eps)
    return (vectors * values.rsqrt().unsqueeze(0)) @ vectors.transpose(0, 1)


def whitened_cross_operator(moments: FMCAMoments, ridge: float = 1e-3) -> Tensor:
    inv_f = _inverse_sqrt(moments.r_f, ridge)
    inv_g = _inverse_sqrt(moments.r_g, ridge)
    return inv_f @ moments.p_fg @ inv_g


def fit_spectral_calibration(
    f: Tensor,
    g_views: Tensor,
    ridge: float = 1e-3,
    centered: bool = True,
) -> SpectralCalibration:
    """Fit whitening and paired singular directions on calibration data.

    Canonical singular values are reported explicitly.  ``eigenvalues`` are
    their squares, so their sum is exactly the trace objective used by the
    experiment plan.
    """

    _check_features(f, g_views)
    mean_f = f.mean(dim=0, keepdim=True) if centered else torch.zeros_like(f[:1])
    mean_g = (
        g_views.mean(dim=(0, 1), keepdim=False).unsqueeze(0)
        if centered
        else torch.zeros_like(g_views[0, :1])
    )
    moments = estimate_moments(f, g_views, centered=centered)
    inv_f = _inverse_sqrt(moments.r_f, ridge)
    inv_g = _inverse_sqrt(moments.r_g, ridge)
    cross = inv_f @ moments.p_fg @ inv_g
    u, singular_values, vh = torch.linalg.svd(cross, full_matrices=False)
    # Row features multiply these matrices on the right.
    transform_f = inv_f @ u
    transform_g = inv_g @ vh.transpose(0, 1)
    eigenvalues = singular_values.square().clamp(min=0.0, max=1.0)
    return SpectralCalibration(
        mean_f=mean_f,
        mean_g=mean_g,
        transform_f=transform_f,
        transform_g=transform_g,
        singular_values=singular_values,
        eigenvalues=eigenvalues,
        ridge=ridge,
        centered=centered,
    )


def calibration_to_state(calibration: SpectralCalibration) -> Dict[str, object]:
    return {
        "mean_f": calibration.mean_f.detach().cpu(),
        "mean_g": calibration.mean_g.detach().cpu(),
        "transform_f": calibration.transform_f.detach().cpu(),
        "transform_g": calibration.transform_g.detach().cpu(),
        "singular_values": calibration.singular_values.detach().cpu(),
        "eigenvalues": calibration.eigenvalues.detach().cpu(),
        "ridge": calibration.ridge,
        "centered": calibration.centered,
    }
