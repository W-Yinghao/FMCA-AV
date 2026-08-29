"""Two-tier finite-sample certificates via matrix Bernstein radii.

The frozen certificate subtracted only the empirical defect, which
implicitly treated the estimated matrices as exact.  The completed
theory supplies explicit high-probability radii for each estimated
matrix under frozen network, coordinates and Gram correction, and
distinguishes two conclusions that were previously conflated:

    Tier 1 (true endpoint)
        [sigma_k(C_comp) - delta_op - r_D]_+  <=  sigma_k(C_dir)

    Tier 2 (population path)
        [sigma_k(C_comp) - delta_op - r_D - 2 r_P]_+
            <=  [sigma_k(C_comp_pop) - ||C_dir_pop - C_comp_pop||_2]_+
        r_P = prod_l (1 + r_l) - 1

Tier 2 is the population path certificate.  Tier 1 alone must never be
captioned as one.  Radii come from matrix Bernstein (Tropp) with a
feature-norm bound registered in the frozen appendum: the empirical
maximum on the Stage-B' split with a 1.5x margin, with clipping
violations counted and reported rather than hidden.
"""

from dataclasses import dataclass
from typing import List, Sequence

import math

import torch
from torch import Tensor

NORM_MARGIN = 1.5
DEFAULT_ALPHA = 0.05


@dataclass
class BernsteinRadius:
    radius: float
    samples: int
    norm_bound: float
    variance: float
    clipped: int

    def as_metrics(self) -> dict:
        return {"radius": self.radius, "samples": self.samples,
                "norm_bound": self.norm_bound, "variance": self.variance,
                "clipped_samples": self.clipped}


def matrix_bernstein_radius(
    left: Tensor, right: Tensor, alpha: float = DEFAULT_ALPHA,
    margin: float = NORM_MARGIN,
) -> BernsteinRadius:
    """High-probability radius for ``E[left right^T]`` estimated by its mean.

    ``left`` is ``[n, d1]`` and ``right`` is ``[n, d2]``; each sample
    contributes the rank-one term ``left_i right_i^T``.  Returns the
    Bernstein radius at joint confidence ``alpha`` for this one matrix;
    callers union-bound across matrices themselves.
    """

    if left.shape[0] != right.shape[0]:
        raise ValueError("left and right must share the sample axis")
    a, b = left.double(), right.double()
    n, d1, d2 = a.shape[0], a.shape[1], b.shape[1]
    if n < 2:
        raise ValueError("at least two samples are required")

    estimate = a.transpose(0, 1) @ b / n
    norms = a.norm(dim=1) * b.norm(dim=1)
    bound = float(norms.max()) * margin
    clipped = int((norms > bound).sum())

    # Per-sample centered terms X_i = a_i b_i^T - estimate; the two
    # one-sided variances are what Bernstein needs.
    gram_a = a.transpose(0, 1) @ (a * (b.norm(dim=1) ** 2).unsqueeze(1)) / n
    gram_b = b.transpose(0, 1) @ (b * (a.norm(dim=1) ** 2).unsqueeze(1)) / n
    variance = max(
        float(torch.linalg.matrix_norm(gram_a - estimate @ estimate.transpose(0, 1), ord=2)),
        float(torch.linalg.matrix_norm(gram_b - estimate.transpose(0, 1) @ estimate, ord=2)),
    )
    variance = max(variance, 0.0)

    log_term = math.log(max((d1 + d2) / alpha, math.e))
    radius = math.sqrt(2.0 * variance * log_term / n) + 2.0 * bound * log_term / (3.0 * n)
    return BernsteinRadius(radius=radius, samples=n, norm_bound=bound,
                           variance=variance, clipped=clipped)


def path_radius(edge_radii: Sequence[float]) -> float:
    """r_P = prod (1 + r_l) - 1, the compounded product-error radius."""

    product = 1.0
    for radius in edge_radii:
        product *= (1.0 + float(radius))
    return product - 1.0


@dataclass
class TieredCertificate:
    tier1: Tensor
    tier2: Tensor
    delta_operator: float
    endpoint_radius: float
    edge_radii: List[float]
    path_radius: float
    alpha: float

    def as_metrics(self, top_k: int = 8) -> dict:
        return {
            "tier1_true_endpoint": [float(v) for v in self.tier1[:top_k]],
            "tier2_population_path": [float(v) for v in self.tier2[:top_k]],
            "tier1_top": float(self.tier1.max()),
            "tier2_top": float(self.tier2.max()),
            "delta_operator": self.delta_operator,
            "endpoint_radius": self.endpoint_radius,
            "edge_radii": [float(v) for v in self.edge_radii],
            "path_radius": self.path_radius,
            "alpha": self.alpha,
        }


def tiered_certificate(
    c_comp: Tensor, delta_operator: float, endpoint_radius: float,
    edge_radii: Sequence[float], alpha: float = DEFAULT_ALPHA,
) -> TieredCertificate:
    """Both tiers from a composed operator and its estimation radii."""

    singular = torch.linalg.svdvals(c_comp.double())
    r_path = path_radius(edge_radii)
    tier1 = (singular - delta_operator - endpoint_radius).clamp_min(0.0)
    tier2 = (singular - delta_operator - endpoint_radius - 2.0 * r_path).clamp_min(0.0)
    return TieredCertificate(
        tier1=tier1, tier2=tier2, delta_operator=float(delta_operator),
        endpoint_radius=float(endpoint_radius),
        edge_radii=[float(v) for v in edge_radii], path_radius=r_path,
        alpha=float(alpha),
    )
