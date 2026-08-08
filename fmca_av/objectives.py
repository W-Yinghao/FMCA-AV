"""Differentiable FMCA-AV dependence objectives."""

import torch
from torch import Tensor

from .operators import FMCAMoments, regularized_covariance, whitened_cross_operator


def trace_score(moments: FMCAMoments, ridge: float = 1e-3) -> Tensor:
    # Centered B-by-K features have rank at most B-1.  Solving the small K-by-K
    # operator in FP64 keeps the configured ridge effective when B is close to K,
    # while the expensive neural forward/backward remains in its trainer precision.
    output_dtype = moments.r_f.dtype
    working_dtype = torch.float64 if output_dtype in {torch.float16, torch.bfloat16, torch.float32} else output_dtype
    r_f = moments.r_f.to(working_dtype)
    r_g = moments.r_g.to(working_dtype)
    p_fg = moments.p_fg.to(working_dtype)
    regularized_f = regularized_covariance(r_f, ridge)
    regularized_g = regularized_covariance(r_g, ridge)
    left = torch.linalg.solve(regularized_f, p_fg)
    right = torch.linalg.solve(regularized_g, p_fg.transpose(0, 1))
    return torch.trace(left @ right).to(output_dtype)


def logdet_score(
    moments: FMCAMoments,
    ridge: float = 1e-3,
    margin: float = 1e-6,
) -> Tensor:
    """Stable ``-log det(I - C C^T)`` form of the block log-det score."""

    if not 0 < margin < 1:
        raise ValueError("margin must lie strictly between zero and one")
    # Match the trace objective's numerical protocol: the K-by-K whitening and
    # spectral solve are inexpensive relative to the network and substantially
    # more reliable in FP64 when covariances contain repeated/near-null modes.
    output_dtype = moments.r_f.dtype
    working_dtype = torch.float64 if output_dtype in {torch.float16, torch.bfloat16, torch.float32} else output_dtype
    working = FMCAMoments(
        moments.r_f.to(working_dtype), moments.r_g.to(working_dtype), moments.p_fg.to(working_dtype),
        moments.count_x, moments.count_y, moments.centered,
    )
    singular_values = torch.linalg.svdvals(whitened_cross_operator(working, ridge=ridge))
    squared = singular_values.square().clamp(max=1.0 - margin)
    return (-torch.log1p(-squared).sum()).to(output_dtype)


def fmca_score(
    moments: FMCAMoments,
    name: str,
    ridge: float = 1e-3,
    logdet_margin: float = 1e-6,
) -> Tensor:
    if name == "trace":
        return trace_score(moments, ridge=ridge)
    if name == "logdet":
        return logdet_score(moments, ridge=ridge, margin=logdet_margin)
    raise ValueError(f"unknown objective {name!r}; expected 'trace' or 'logdet'")
