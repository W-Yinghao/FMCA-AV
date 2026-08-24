"""Gram correction: the orthogonal-projection defect, not the ridge surrogate.

With ridge whitening ``W_l = (R_l + rho s_l I)^{-1/2}`` the coordinate
functions at stage ``l`` have Gram matrix ``G_l = W_l R_l W_l != I``, so
multiplying whitened cross-matrices directly inserts ``F_l F_l^*`` at
every interior interface -- which is not an orthogonal projection.  The
error splits exactly:

    I - F_l F_l^*  =  (I - P_l)  +  Q_l (I - G_l) Q_l^*
    P_l = F_l G_l^{-1} F_l^*,    Q_l = F_l G_l^{-1/2}

The first term is the finite-subspace omission the paper's theorems are
about; the second is a coordinate-metric artifact.  Everything the
pipeline reported before this module measured their sum.

The corrected quantities insert the Gram inverses:

    C_comp = G_0^{-1/2} B_{0,1} G_1^{-1} B_{1,2} ... G_{L-1}^{-1} B_{L-1,L} G_L^{-1/2}
    C_dir  = G_0^{-1/2} B_{0,L} G_L^{-1/2}

Inversion is spectral and truncated at ``tau * lambda_max`` (registered
in the frozen appendum as 1e-3); the retained rank per level is part of
the report, never silently dropped.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor

TAU_RELATIVE = 1e-3


def gram_matrix(encoded: Tensor) -> Tensor:
    """Second moment of already-encoded (whitened) coordinate features.

    Accepts ``[samples, dim]`` or ``[parents, views, dim]``; views are
    pooled, since the Gram is a property of the coordinate system rather
    than of any one view.
    """

    flat = encoded.flatten(0, -2) if encoded.ndim > 2 else encoded
    if flat.ndim != 2:
        raise ValueError(f"expected [.., dim] features, got {tuple(encoded.shape)}")
    if flat.shape[0] < 2:
        raise ValueError("at least two samples are required to estimate a Gram matrix")
    work = flat.double()
    gram = work.transpose(0, 1) @ work / work.shape[0]
    return 0.5 * (gram + gram.transpose(0, 1))


def spectral_pseudo_inverse(
    gram: Tensor, power: float = -1.0, tau_relative: float = TAU_RELATIVE
) -> Tuple[Tensor, int]:
    """``G^power`` on the retained eigen-subspace, plus the retained rank.

    ``power`` is -1 for interior interfaces and -0.5 for the two ends.
    Eigenvalues below ``tau_relative * lambda_max`` are dropped rather
    than inverted, which is what keeps the correction from amplifying
    directions the ridge already destroyed.
    """

    values, vectors = torch.linalg.eigh(gram.double())
    largest = float(values.max())
    if largest <= 0:
        raise ValueError("Gram matrix has no positive eigenvalue")
    keep = values >= tau_relative * largest
    retained = int(keep.sum())
    if retained == 0:
        raise ValueError("Gram truncation retained no directions")
    kept_values = values[keep]
    kept_vectors = vectors[:, keep]
    scaled = kept_values.pow(power)
    return (kept_vectors * scaled) @ kept_vectors.transpose(0, 1), retained


@dataclass
class GramCorrection:
    """Per-level Gram matrices and the inverses the composition needs."""

    grams: List[Tensor]
    interior_inverses: List[Tensor]
    end_inverse_sqrts: Tuple[Tensor, Tensor]
    retained_ranks: List[int]
    metric_deviation: List[float]
    tau_relative: float

    def as_metrics(self) -> dict:
        return {
            "retained_ranks": list(self.retained_ranks),
            "metric_deviation": [float(value) for value in self.metric_deviation],
            "tau_relative": float(self.tau_relative),
            "dimension": int(self.grams[0].shape[0]),
        }


def build_correction(
    grams: Sequence[Tensor], tau_relative: float = TAU_RELATIVE
) -> GramCorrection:
    """Assemble every inverse the corrected composition needs, once."""

    if len(grams) < 2:
        raise ValueError("a chain needs at least two levels")
    identity = torch.eye(grams[0].shape[0], dtype=torch.float64)
    deviation = [float(torch.linalg.matrix_norm(g.double() - identity, ord=2)) for g in grams]
    ranks: List[int] = []
    interior: List[Tensor] = []
    for level, gram in enumerate(grams):
        if 0 < level < len(grams) - 1:
            inverse, rank = spectral_pseudo_inverse(gram, -1.0, tau_relative)
            interior.append(inverse)
        else:
            _, rank = spectral_pseudo_inverse(gram, -0.5, tau_relative)
        ranks.append(rank)
    first, _ = spectral_pseudo_inverse(grams[0], -0.5, tau_relative)
    last, _ = spectral_pseudo_inverse(grams[-1], -0.5, tau_relative)
    return GramCorrection(
        grams=[g.double() for g in grams],
        interior_inverses=interior,
        end_inverse_sqrts=(first, last),
        retained_ranks=ranks,
        metric_deviation=deviation,
        tau_relative=float(tau_relative),
    )


def corrected_composition(edges: Sequence[Tensor], correction: GramCorrection) -> Tensor:
    """G_0^{-1/2} B_01 G_1^{-1} B_12 ... G_{L-1}^{-1} B_{L-1,L} G_L^{-1/2}."""

    if len(edges) != len(correction.grams) - 1:
        raise ValueError(
            f"{len(edges)} edges need {len(edges) + 1} levels, got {len(correction.grams)}"
        )
    first, last = correction.end_inverse_sqrts
    product = first @ edges[0].double()
    for index, edge in enumerate(edges[1:]):
        product = product @ correction.interior_inverses[index] @ edge.double()
    return product @ last


def corrected_endpoint(c_dir: Tensor, correction: GramCorrection) -> Tensor:
    """G_0^{-1/2} B_{0,L} G_L^{-1/2}."""

    first, last = correction.end_inverse_sqrts
    return first @ c_dir.double() @ last


def cumulative_interface_attribution(
    edges: Sequence[Tensor], c_dir: Tensor, correction: GramCorrection
) -> List[dict]:
    """Defect with corrections switched on one interface at a time.

    Attributes the surrogate-to-projection gap per interface, which is
    what says how much of a previously reported defect was ridge
    artifact.  These are interface leakage DIAGNOSTICS: the theory notes
    establish the telescoping terms can cancel and depend on expansion
    direction, so they carry no causal per-layer attribution.
    """

    identity = torch.eye(correction.grams[0].shape[0], dtype=torch.float64)
    steps = []
    for enabled in range(len(correction.interior_inverses) + 1):
        partial = GramCorrection(
            grams=correction.grams,
            interior_inverses=(correction.interior_inverses[:enabled]
                               + [identity] * (len(correction.interior_inverses) - enabled)),
            end_inverse_sqrts=(correction.end_inverse_sqrts if enabled else (identity, identity)),
            retained_ranks=correction.retained_ranks,
            metric_deviation=correction.metric_deviation,
            tau_relative=correction.tau_relative,
        )
        comp = corrected_composition(edges, partial)
        direct = corrected_endpoint(c_dir, partial)
        delta = direct - comp
        steps.append({
            "interfaces_corrected": enabled,
            "delta_frobenius": float(torch.linalg.matrix_norm(delta, ord="fro")),
            "delta_operator": float(torch.linalg.matrix_norm(delta, ord=2)),
            "dir_frobenius": float(torch.linalg.matrix_norm(direct, ord="fro")),
        })
    return steps
