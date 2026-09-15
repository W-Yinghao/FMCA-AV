"""Control battery for the path-supported certificate.

Positive controls (report quantities MUST be invariant):
  - two-sided gauge rotation of an interior interface (Prop: gauge invariance)

Negative controls (report quantities MUST degrade / change):
  - one-sided interface rotation
  - parent-child pairing shuffle (noise floor)
  - endpoint pairing shuffle
  - layer-order shuffle of the composition
  - centering off (constant-mode pollution)
  - edge-wise independent calibration instead of shared level coordinates
  - naive per-edge singular-value product (in triplet.naive_singular_value_product)
"""

from typing import List, Optional, Sequence

import torch
from torch import Tensor

from .coordinates import LevelCoordinates, fit_level_coordinates
from .estimation import (
    ChainFeatureBatch,
    conditional_cross_operator,
    estimate_edge_operators,
    level_calibration_features,
)


def random_orthogonal(
    dim: int,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Haar-distributed orthogonal matrix via QR with sign correction."""

    raw = torch.randn(dim, dim, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(raw)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def rotate_interface(
    edges: Sequence[Tensor],
    interface: int,
    rotation: Tensor,
    sides: str = "both",
) -> List[Tensor]:
    """Rotate the shared basis at interior interface ``interface``.

    Interface j sits between edge j-1 (incoming) and edge j (outgoing).
    ``sides='both'`` applies C_{j-1} -> C_{j-1} Q and C_j -> Q^T C_j, a pure
    re-parameterization of the interface basis: the composition must be
    invariant.  ``sides='left'`` or ``'right'`` rotates only one factor and
    must change the composition (negative control).
    """

    if not 1 <= interface <= len(edges) - 1:
        raise ValueError("interface must be an interior interface index")
    incoming = edges[interface - 1]
    outgoing = edges[interface]
    if rotation.shape != (incoming.shape[1], incoming.shape[1]):
        raise ValueError("rotation must match the interface dimension")
    if outgoing.shape[0] != incoming.shape[1]:
        raise ValueError("edges do not compose at this interface")
    rotated = [edge.clone() for edge in edges]
    if sides == "both":
        rotated[interface - 1] = incoming @ rotation
        rotated[interface] = rotation.transpose(0, 1) @ outgoing
    elif sides == "left":
        rotated[interface - 1] = incoming @ rotation
    elif sides == "right":
        rotated[interface] = rotation.transpose(0, 1) @ outgoing
    else:
        raise ValueError("sides must be both, left, or right")
    return rotated


def shuffle_edge_order(edges: Sequence[Tensor], permutation: Sequence[int]) -> List[Tensor]:
    """Reorder the composition (negative control).

    Requires the permuted sequence to remain dimensionally composable, which
    holds automatically when all interface dimensions are equal.
    """

    if sorted(permutation) != list(range(len(edges))):
        raise ValueError("permutation must reorder exactly the edge indices")
    reordered = [edges[index] for index in permutation]
    for left, right in zip(reordered[:-1], reordered[1:]):
        if left.shape[1] != right.shape[0]:
            raise ValueError("permuted edges are not composable; use equal interface dimensions")
    return reordered


def shuffle_parent_pairing(
    parent: Tensor,
    children: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Break the parent-child pairing and re-estimate the edge operator."""

    permutation = torch.randperm(parent.shape[0], generator=generator)
    return conditional_cross_operator(parent[permutation], children)


def pairing_noise_floor(
    parent: Tensor,
    children: Tensor,
    repeats: int = 20,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Frobenius norms of pairing-shuffled operators: the noise floor that a
    claimed dependence must clearly exceed."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    norms = [
        torch.linalg.matrix_norm(shuffle_parent_pairing(parent, children, generator), ord="fro")
        for _ in range(repeats)
    ]
    return torch.stack(norms)


def edgewise_calibrated_edges(
    batch: ChainFeatureBatch,
    calibration: ChainFeatureBatch,
    ridge: float = 1e-3,
) -> List[Tensor]:
    """FORBIDDEN design, kept as a negative control: each edge fits its own
    independent coordinates for its two endpoints instead of sharing one
    coordinate system per level.

    The parent side of edge l is calibrated on chain states only, while the
    child side is calibrated on descendant samples only, so the interface at
    level l no longer has a single basis serving the incoming and outgoing
    edges.  Composing these operators is exactly the mistake the shared
    Stage-B protocol forbids.
    """

    edges: List[Tensor] = []
    for edge in range(batch.num_levels - 1):
        parent_coordinates = fit_level_coordinates(calibration.chain[edge], ridge=ridge)
        child_coordinates = fit_level_coordinates(calibration.children[edge], ridge=ridge)
        parent = parent_coordinates.encode(batch.chain[edge])
        children = child_coordinates.encode(batch.children[edge])
        edges.append(conditional_cross_operator(parent, children))
    return edges


def uncentered_coordinates(
    calibration: ChainFeatureBatch,
    ridge: float = 1e-3,
) -> List[LevelCoordinates]:
    """Centering-off coordinates (negative control): keeps the constant mode,
    which manufactures a trivial near-unit singular direction on every edge
    whenever the level means are nonzero."""

    return [
        fit_level_coordinates(level_calibration_features(calibration, level), ridge=ridge, centered=False)
        for level in range(calibration.num_levels)
    ]


def endpoint_pairing_shuffle(
    batch: ChainFeatureBatch,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Endpoint operator with the root-leaf pairing destroyed (noise floor)."""

    permutation = torch.randperm(batch.num_parents, generator=generator)
    leaf = (
        batch.endpoint_descendants
        if batch.endpoint_descendants is not None
        else batch.chain[-1].unsqueeze(1)
    )
    return conditional_cross_operator(batch.chain[0][permutation], leaf)


def layer_order_shuffled_composition(
    batch: ChainFeatureBatch,
    permutation: Sequence[int],
) -> List[Tensor]:
    """Estimate edges normally, then compose them in a shuffled order."""

    return shuffle_edge_order(estimate_edge_operators(batch), permutation)
