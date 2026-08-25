"""ChainSpec: one correctness engine, many applications.

The strategy document's §6 asks every application to declare the same
six things rather than growing its own ad-hoc pipeline: the state chain,
the transitions, the feature taps, the coordinate rule, the split rule,
and what the measurement is allowed to mean.  This module is that
contract plus the shared estimator, so a new pilot supplies states and
an interpretation and inherits the object correctness.

Hard constraints enforced here, not left to the caller:

- features are centered (a constant mode manufactures perfect transport
  on every edge);
- one intermediate state has exactly ONE coordinate calibration, shared
  by the edge on its left and the edge on its right;
- interior interfaces carry Gram inverses (or the caller re-orthonormalizes
  exactly, which the engine checks agrees);
- full matrices are multiplied before any SVD -- never per-mode products;
- endpoint, path and discrepancy are reported jointly with ranks,
  conditioning and N.

Deterministic transitions are first-class: for ``H_{l+1} = F_l(H_l)``
the conditional-expectation operator is still a tower-property
composition, so a pretrained activation chain is a legitimate ChainSpec
even though no view is resampled.  The finite projected defect then
measures whether the chosen coordinates are sufficient for the suffix.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor

from .coordinates import LevelCoordinates, fit_level_coordinates
from .finite_sample import matrix_bernstein_radius, path_radius
from .gram import (
    build_correction,
    corrected_composition,
    corrected_endpoint,
    cumulative_interface_attribution,
    gram_matrix,
)
from .triplet import certificate_report, compose_edge_operators

INTERPRETATIONS = {
    "depth_sufficiency",       # deterministic activation chain
    "statistical_stitchability",  # U -> V -> Y bridge, NOT a physical adapter
    "view_composability",      # stochastic view channel at fixed depth
    "corruption_path",         # recursive corruption chain
}


@dataclass
class ChainSpec:
    """What a study must declare before it may be measured."""

    name: str
    interpretation: str
    state_names: List[str]
    deterministic_edges: bool
    coordinate_budget: Optional[int] = None
    gram_bound: Optional[float] = None
    coordinate_ridge: float = 1e-3
    retained_rank_rule: str = "tau=1e-3*lambda_max"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.interpretation not in INTERPRETATIONS:
            raise ValueError(
                f"interpretation must be one of {sorted(INTERPRETATIONS)}; "
                f"an undeclared interpretation is how a statistical bridge gets "
                f"called a physical adapter"
            )
        if len(self.state_names) < 3:
            raise ValueError("a path needs at least two edges, so at least three states")
        if self.coordinate_budget is not None and self.coordinate_budget < 2:
            raise ValueError("a coordinate budget below 2 leaves no operator to measure")

    def as_metrics(self) -> dict:
        return {
            "name": self.name,
            "interpretation": self.interpretation,
            "states": list(self.state_names),
            "deterministic_edges": self.deterministic_edges,
            "coordinate_budget": self.coordinate_budget,
            "gram_bound": self.gram_bound,
            "coordinate_ridge": self.coordinate_ridge,
            "retained_rank_rule": self.retained_rank_rule,
            "notes": self.notes,
        }


def _centered_cross(left: Tensor, right: Tensor) -> Tensor:
    """E[z_left z_right^T] on already-encoded (hence centered) features."""

    if left.shape[0] != right.shape[0]:
        raise ValueError("cross-moment operands must share the sample axis")
    return left.double().transpose(0, 1) @ right.double() / left.shape[0]


def _budget_basis(state: Tensor, budget: Optional[int], gram_bound: Optional[float],
                  ridge: float):
    """The retained principal directions of one state, or None if unneeded.

    Fitted on CALIBRATION only and then frozen.  Two rules, two jobs:

    ``budget`` matches the number of coordinates across levels, because a
    chain whose levels are 64 and 2048 wide otherwise confounds depth
    with width.

    ``gram_bound`` is the conditioning guard, stated as the thing it
    guarantees.  Ridge whitening leaves Gram eigenvalues
    ``lambda / (lambda + penalty)``, so keeping only directions with
    ``lambda * f >= (1 - f) * penalty`` bounds the level's ``||I - G||``
    by ``f``.  The penalty scale is anchored to the NATIVE feature space
    (rho times the native mean marginal variance) and carried through
    the projection, because letting the projection re-derive the scale
    from its own retained spectrum couples the threshold to the very
    directions it is judging -- a one-huge-direction spectrum like
    ConvNeXt's then inflates its own ridge and disqualifies a tail that
    is in fact perfectly conditioned under the ridge the full feature
    space induces.
    """

    if budget is None and gram_bound is None:
        return None
    work = state.double()
    centered = work - work.mean(0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered / centered.shape[0]
    values, vectors = torch.linalg.eigh(0.5 * (covariance + covariance.transpose(0, 1)))
    order = torch.argsort(values, descending=True)
    values, vectors = values[order].clamp_min(0.0), vectors[:, order]
    native_scale = float(values.mean())  # tr / d on the native features
    keep = values.shape[0]
    if gram_bound is not None:
        fraction = float(gram_bound)
        if not 0.0 < fraction < 1.0:
            raise ValueError("gram_bound must lie strictly between 0 and 1")
        threshold = (1.0 - fraction) / fraction * ridge * native_scale
        keep = int((values >= threshold).sum())
    if budget is not None:
        keep = min(keep, int(budget))
    if keep < 2:
        raise ValueError(
            f"coordinate rule retained {keep} directions of {state.shape[-1]}; "
            f"nothing is left to measure"
        )
    kept_trace = float(values[:keep].sum())
    total = float(values.sum())
    if keep == state.shape[-1]:
        return None
    # fit_level_coordinates applies penalty = ridge_arg * (kept trace / keep);
    # this multiplier restores the native-anchored penalty rho * native_scale.
    ridge_effective = ridge * native_scale * keep / kept_trace
    return vectors[:, :keep], (kept_trace / total if total > 0 else 0.0), ridge_effective


def _apply_budget(states: Sequence[Tensor], bases) -> List[Tensor]:
    return [state.double() if basis is None else state.double() @ basis[0]
            for state, basis in zip(states, bases)]


def _level_ridges(bases, ridge: float) -> List[float]:
    return [ridge if basis is None else basis[2] for basis in bases]


def measure_chain(
    calibration_states: Sequence[Tensor],
    evaluation_states: Sequence[Tensor],
    spec: ChainSpec,
    top_k: int = 8,
    with_radii: bool = False,
) -> dict:
    """The full multi-component readout for one chain.

    ``calibration_states`` fits coordinates and Grams (Stage B);
    ``evaluation_states`` supplies the operators (Stage C).  They must be
    disjoint samples -- the caller owns the split rule, the engine
    records the sizes so an undisclosed coupling is visible.
    """

    if len(calibration_states) != len(spec.state_names):
        raise ValueError("one calibration tensor per declared state")
    if len(evaluation_states) != len(spec.state_names):
        raise ValueError("one evaluation tensor per declared state")

    native_dimensions = [int(state.shape[-1]) for state in calibration_states]
    bases = [_budget_basis(state, spec.coordinate_budget, spec.gram_bound,
                           spec.coordinate_ridge)
             for state in calibration_states]
    calibration_states = _apply_budget(calibration_states, bases)
    evaluation_states = _apply_budget(evaluation_states, bases)
    retained_variance = [1.0 if basis is None else basis[1] for basis in bases]

    ridges = _level_ridges(bases, spec.coordinate_ridge)
    coordinates: List[LevelCoordinates] = [
        fit_level_coordinates(state, ridge=level_ridge, centered=True)
        for state, level_ridge in zip(calibration_states, ridges)
    ]
    # One calibration per state, reused on both of its edges: the shared
    # coordinate rule is structural here rather than a convention.
    grams = [gram_matrix(coordinates[i].encode(calibration_states[i]))
             for i in range(len(coordinates))]
    correction = build_correction(grams)

    encoded = [coordinates[i].encode(evaluation_states[i])
               for i in range(len(coordinates))]
    edges = [_centered_cross(encoded[i], encoded[i + 1]) for i in range(len(encoded) - 1)]
    c_dir = _centered_cross(encoded[0], encoded[-1])

    surrogate = certificate_report(c_dir, edges=edges, top_k=top_k)
    projected_comp = corrected_composition(edges, correction)
    projected_dir = corrected_endpoint(c_dir, correction)
    projected = certificate_report(projected_dir, c_comp=projected_comp, top_k=top_k)

    def readout(report) -> dict:
        endpoint = report.endpoint_singular_values.double()
        path = report.path_singular_values.double()
        return {
            "endpoint_top": float(endpoint.max()),
            "endpoint_mass": float(endpoint.sum()),
            "endpoint_effective_rank": _effective_rank(endpoint),
            "path_top": float(path.max()),
            "path_mass": float(path.sum()),
            "delta_frobenius": report.delta_frobenius,
            "delta_operator": report.delta_operator,
            "alignment": report.alignment,
            "certified_top": float(report.certified_spectrum.max()),
        }

    finite = None
    if with_radii:
        # Bernstein radii for every estimated matrix, and the compounded
        # path radius the Tier-2 certificate subtracts.  Estimated on the
        # same evaluation split the operators came from.
        edge_radii = [matrix_bernstein_radius(encoded[i], encoded[i + 1]).as_metrics()
                      for i in range(len(encoded) - 1)]
        endpoint = matrix_bernstein_radius(encoded[0], encoded[-1]).as_metrics()
        finite = {
            "edge_radii": edge_radii,
            "endpoint_radius": endpoint,
            "path_radius": path_radius([e["radius"] for e in edge_radii]),
        }

    return {
        "spec": spec.as_metrics(),
        "finite_sample": finite,
        "projection": readout(projected),
        "surrogate": readout(surrogate),
        "gram": correction.as_metrics(),
        "interface_attribution": cumulative_interface_attribution(edges, c_dir, correction),
        "calibration_samples": int(calibration_states[0].shape[0]),
        "evaluation_samples": int(evaluation_states[0].shape[0]),
        "dimensions": [int(state.shape[-1]) for state in calibration_states],
        "native_dimensions": native_dimensions,
        "budget_retained_variance": retained_variance,
    }


def _effective_rank(values: Tensor) -> float:
    positive = values[values > 0]
    if positive.numel() == 0:
        return 0.0
    probabilities = positive / positive.sum()
    entropy = float(-(probabilities * probabilities.log()).sum())
    return float(torch.exp(torch.tensor(entropy)))


def shuffled_null(
    calibration_states: Sequence[Tensor],
    evaluation_states: Sequence[Tensor],
    spec: ChainSpec,
    seed: int = 0,
) -> dict:
    """The pairing-shuffled noise floor for this chain.

    Breaking the sample pairing at the endpoint destroys every genuine
    dependence while leaving dimensions, ranks and sample sizes intact,
    so it says what the estimator reports when there is nothing to
    report.
    """

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(evaluation_states[-1].shape[0], generator=generator)
    shuffled = list(evaluation_states[:-1]) + [evaluation_states[-1][permutation]]
    return measure_chain(calibration_states, shuffled, spec)
