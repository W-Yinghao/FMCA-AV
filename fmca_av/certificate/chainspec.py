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

    def as_metrics(self) -> dict:
        return {
            "name": self.name,
            "interpretation": self.interpretation,
            "states": list(self.state_names),
            "deterministic_edges": self.deterministic_edges,
            "coordinate_ridge": self.coordinate_ridge,
            "retained_rank_rule": self.retained_rank_rule,
            "notes": self.notes,
        }


def _centered_cross(left: Tensor, right: Tensor) -> Tensor:
    """E[z_left z_right^T] on already-encoded (hence centered) features."""

    if left.shape[0] != right.shape[0]:
        raise ValueError("cross-moment operands must share the sample axis")
    return left.double().transpose(0, 1) @ right.double() / left.shape[0]


def measure_chain(
    calibration_states: Sequence[Tensor],
    evaluation_states: Sequence[Tensor],
    spec: ChainSpec,
    top_k: int = 8,
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

    coordinates: List[LevelCoordinates] = [
        fit_level_coordinates(state, ridge=spec.coordinate_ridge, centered=True)
        for state in calibration_states
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

    return {
        "spec": spec.as_metrics(),
        "projection": readout(projected),
        "surrogate": readout(surrogate),
        "gram": correction.as_metrics(),
        "interface_attribution": cumulative_interface_attribution(edges, c_dir, correction),
        "calibration_samples": int(calibration_states[0].shape[0]),
        "evaluation_samples": int(evaluation_states[0].shape[0]),
        "dimensions": [int(state.shape[-1]) for state in calibration_states],
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
