"""FMCA-AV moment operators and finite-dimensional spectral calibration.

The implementation follows equations (2)--(4) of the paper.  In particular,
``R_g`` is the mean of per-view outer products.  Only the cross moment uses the
conditional view mean.  Centering is enabled by default to remove the constant
mode before calibration, as required by the experiment plan.
"""

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor


SCIENTIFIC_CORRECTNESS_VERSION = "20260809_scientific_correctness_v1"


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
        logdet, clipped = clipped_logdet_from_eigenvalues(eigenvalues, margin=1e-7)
        return {
            "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
            "centered": self.centered,
            "ridge": self.ridge,
            "ridge_mode": "relative_to_mean_marginal_variance",
            "singular_values": singular_values.tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "trace_score": float(eigenvalues.sum()),
            "logdet_score": float(logdet),
            "logdet_clipped_mode_count": clipped,
        }


@dataclass
class HeldOutSpectrum:
    """Spectrum of the full held-out canonical cross-covariance matrix."""

    cross_operator: Tensor
    singular_values: Tensor
    eigenvalues: Tensor
    diagonal_correlations: Tensor


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


def _require_finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} contains non-finite values")


def relative_ridge_scale(matrix: Tensor) -> Tensor:
    """Return mean marginal variance with only a dtype-safe zero floor.

    For every finite, nonzero covariance ``R`` and scalar ``a``, this scale
    obeys ``scale(a**2 * R) == a**2 * scale(R)`` up to floating-point error.
    The ``dtype.tiny`` floor is used only for an exactly zero or subnormal
    marginal scale; unlike a unit floor, it does not turn ordinary small-scale
    features into an absolute-ridge objective.
    """

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not matrix.is_floating_point():
        raise TypeError("covariance must have a floating-point dtype")
    _require_finite("covariance", matrix)
    scale = matrix.diagonal().abs().mean().detach()
    return scale.clamp_min(torch.finfo(matrix.dtype).tiny)


def regularized_covariance(matrix: Tensor, ridge: float) -> Tensor:
    """Apply the project's relative ridge rule to a covariance matrix."""

    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    scale = relative_ridge_scale(matrix)
    # Ensure an exactly zero covariance remains invertible.  This lower bound
    # activates only at the representable limit of the working dtype.
    penalty = (scale * ridge).clamp_min(torch.finfo(matrix.dtype).tiny)
    dimension = matrix.shape[0]
    return matrix + penalty * torch.eye(
        dimension, dtype=matrix.dtype, device=matrix.device
    )


def inverse_sqrt_covariance(matrix: Tensor, ridge: float) -> Tensor:
    """Return the inverse square root under the shared relative-ridge rule."""

    regularized = regularized_covariance(matrix, ridge)
    values, vectors = torch.linalg.eigh(regularized)
    values = values.clamp_min(torch.finfo(matrix.dtype).tiny)
    return (vectors * values.rsqrt().unsqueeze(0)) @ vectors.transpose(0, 1)


def whitened_cross_operator(moments: FMCAMoments, ridge: float = 1e-3) -> Tensor:
    inv_f = inverse_sqrt_covariance(moments.r_f, ridge)
    inv_g = inverse_sqrt_covariance(moments.r_g, ridge)
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
    inv_f = inverse_sqrt_covariance(moments.r_f, ridge)
    inv_g = inverse_sqrt_covariance(moments.r_g, ridge)
    cross = inv_f @ moments.p_fg @ inv_g
    u, singular_values, vh = torch.linalg.svd(cross, full_matrices=False)
    # Row features multiply these matrices on the right.
    transform_f = inv_f @ u
    transform_g = inv_g @ vh.transpose(0, 1)
    # Preserve the raw finite-sample spectrum.  Values near or above one are
    # clipped only by an explicit logdet/TSD calculation, never at calibration.
    eigenvalues = singular_values.square()
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
        "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
        "mean_f": calibration.mean_f.detach().cpu(),
        "mean_g": calibration.mean_g.detach().cpu(),
        "transform_f": calibration.transform_f.detach().cpu(),
        "transform_g": calibration.transform_g.detach().cpu(),
        "singular_values": calibration.singular_values.detach().cpu(),
        "eigenvalues": calibration.eigenvalues.detach().cpu(),
        "ridge": calibration.ridge,
        "centered": calibration.centered,
    }


CalibrationLike = Union[SpectralCalibration, Mapping[str, object]]


def _calibration_tensor(calibration: CalibrationLike, name: str, reference: Tensor) -> Tensor:
    value = getattr(calibration, name) if isinstance(calibration, SpectralCalibration) else calibration[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"calibration {name} must be a tensor")
    result = value.to(device=reference.device, dtype=reference.dtype)
    _require_finite(f"calibration {name}", result)
    return result


def evaluate_heldout_spectrum(
    f: Tensor,
    g_views: Tensor,
    calibration: CalibrationLike,
) -> HeldOutSpectrum:
    """Evaluate the full held-out canonical cross operator.

    Calibration means and transforms remain fixed.  The returned singular
    values come from the complete matrix
    ``C_test = z_f.T @ mean_views(z_g) / number_of_parents``.  Its diagonal is
    retained only as an explicitly named diagnostic and is not a spectrum.
    """

    _check_features(f, g_views)
    _require_finite("held-out f", f)
    _require_finite("held-out g_views", g_views)
    mean_f = _calibration_tensor(calibration, "mean_f", f)
    mean_g = _calibration_tensor(calibration, "mean_g", g_views)
    transform_f = _calibration_tensor(calibration, "transform_f", f)
    transform_g = _calibration_tensor(calibration, "transform_g", g_views)
    z_f = (f - mean_f) @ transform_f
    z_g = (g_views - mean_g) @ transform_g
    cross = z_f.transpose(0, 1) @ z_g.mean(dim=1) / f.shape[0]
    singular_values = torch.linalg.svdvals(cross)
    return HeldOutSpectrum(
        cross_operator=cross,
        singular_values=singular_values,
        eigenvalues=singular_values.square(),
        diagonal_correlations=torch.diagonal(cross),
    )


def clipped_logdet_from_eigenvalues(
    eigenvalues: Tensor,
    margin: float = 1e-7,
) -> Tuple[Tensor, int]:
    """Compute a stable logdet/TSD while preserving the caller's raw values."""

    if not 0 < margin < 1:
        raise ValueError("margin must lie strictly between zero and one")
    _require_finite("eigenvalues", eigenvalues)
    if bool((eigenvalues.detach() < 0).any()):
        raise ValueError("eigenvalues must be non-negative")
    threshold = 1.0 - margin
    clipped_count = int((eigenvalues.detach() >= threshold).sum().cpu())
    bounded = eigenvalues.clamp(max=threshold)
    return -torch.log1p(-bounded).sum(), clipped_count


def dependence_contribution_maps(
    parent_canonical: Tensor,
    local_canonical: Tensor,
    singular_values: Tensor,
    modes: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Compute local paired canonical dependence contributions.

    For parent coordinates ``u`` and local child coordinates ``v_p``, the
    signed contribution is ``D(p) = sum_k s_k u_k v_{p,k}``.  Its absolute
    value is the default non-negative localization map.  The former g-only
    energy is returned solely as the explicitly named baseline
    ``sum_k s_k**2 v_{p,k}**2``.
    """

    if parent_canonical.ndim == 2 and parent_canonical.shape[0] == 1:
        parent_canonical = parent_canonical[0]
    if parent_canonical.ndim != 1:
        raise ValueError("parent_canonical must have shape [modes] or [1, modes]")
    if local_canonical.ndim != 2:
        raise ValueError("local_canonical must have shape [positions, modes]")
    if singular_values.ndim != 1:
        raise ValueError("singular_values must have shape [modes]")
    for name, value in (
        ("parent_canonical", parent_canonical),
        ("local_canonical", local_canonical),
        ("singular_values", singular_values),
    ):
        _require_finite(name, value)
    available = min(parent_canonical.shape[0], local_canonical.shape[1], singular_values.shape[0])
    count = available if modes is None else min(available, int(modes))
    if count < 1:
        raise ValueError("at least one canonical mode is required")
    weights = singular_values[:count]
    local = local_canonical[:, :count]
    signed = (local * (parent_canonical[:count] * weights)).sum(dim=1)
    return {
        "signed_dependence": signed,
        "absolute_dependence": signed.abs(),
        "g_energy_baseline": (local.square() * weights.square()).sum(dim=1),
    }
