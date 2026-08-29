"""Stage-C operator estimation in frozen Stage-B coordinates.

The data unit is a nested-chain feature batch: realized chain states per
level, plus M conditionally independent descendants of each realized parent
per edge, plus (optionally) full-path descendants of the root for the
endpoint operator.  All operators -- adjacent and endpoint -- are evaluated
in the SAME frozen per-level coordinates.

Terminology discipline: estimates from a shared sample are "separately
estimated"; only estimates from disjoint splits (``split_chain_batch``) may
be called cross-fitted.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .coordinates import LevelCoordinates


@dataclass
class ChainFeatureBatch:
    """Features of one nested-chain sample batch.

    chain[l]:    [N, K_l]      realized chain state at level l (l = 0..L)
    children[l]: [N, M_l, K_{l+1}]  conditional descendants of chain[l] (l = 0..L-1)
    endpoint_descendants: [N, M, K_L] optional full-path descendants of chain[0]
    """

    chain: List[Tensor]
    children: List[Tensor]
    endpoint_descendants: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if len(self.chain) < 2:
            raise ValueError("a chain needs at least two levels")
        if len(self.children) != len(self.chain) - 1:
            raise ValueError("children must hold exactly one entry per edge")
        parents = self.chain[0].shape[0]
        if parents < 2:
            raise ValueError("a chain batch needs at least two parents")
        for level, states in enumerate(self.chain):
            if states.ndim != 2 or states.shape[0] != parents:
                raise ValueError(f"chain level {level} must be [N, K], got {tuple(states.shape)}")
        for edge, descendants in enumerate(self.children):
            if descendants.ndim != 3 or descendants.shape[0] != parents or descendants.shape[1] < 1:
                raise ValueError(
                    f"children of edge {edge} must be [N, M>=1, K], got {tuple(descendants.shape)}"
                )
            if descendants.shape[2] != self.chain[edge + 1].shape[1]:
                raise ValueError(
                    f"children of edge {edge} must live at level {edge + 1} "
                    f"({self.chain[edge + 1].shape[1]} dims), got {descendants.shape[2]}"
                )
        if self.endpoint_descendants is not None:
            leaf_dim = self.chain[-1].shape[1]
            if (
                self.endpoint_descendants.ndim != 3
                or self.endpoint_descendants.shape[0] != parents
                or self.endpoint_descendants.shape[1] < 1
                or self.endpoint_descendants.shape[2] != leaf_dim
            ):
                raise ValueError("endpoint_descendants must be [N, M>=1, K_L]")

    @property
    def num_parents(self) -> int:
        return int(self.chain[0].shape[0])

    @property
    def num_levels(self) -> int:
        return len(self.chain)


def level_calibration_features(batch: ChainFeatureBatch, level: int) -> Tensor:
    """All available level samples for Stage-B fitting: chain states plus
    the flattened descendants that live at this level."""

    parts = [batch.chain[level]]
    if level >= 1:
        parts.append(batch.children[level - 1].flatten(0, 1))
    return torch.cat(parts, dim=0)


def encode_chain_batch(
    batch: ChainFeatureBatch,
    coordinates: Sequence[LevelCoordinates],
) -> ChainFeatureBatch:
    """Map a raw batch into the frozen shared coordinates, level by level."""

    if len(coordinates) != batch.num_levels:
        raise ValueError("one coordinate system per level is required")
    chain = [coordinates[level].encode(states) for level, states in enumerate(batch.chain)]
    children = [
        coordinates[edge + 1].encode(descendants) for edge, descendants in enumerate(batch.children)
    ]
    endpoint = (
        coordinates[-1].encode(batch.endpoint_descendants)
        if batch.endpoint_descendants is not None
        else None
    )
    return ChainFeatureBatch(chain=chain, children=children, endpoint_descendants=endpoint)


def conditional_cross_operator(parent: Tensor, children: Tensor) -> Tensor:
    """C = (1/N) sum_i z_parent(i) (mean_m z_child(i, m))^T in frozen coordinates.

    The child view mean is the Monte Carlo estimate of E[z_child | parent].
    No additional centering or whitening happens here: coordinates are frozen
    at Stage B and shared with every other edge touching these levels.
    """

    if parent.ndim != 2 or children.ndim != 3 or parent.shape[0] != children.shape[0]:
        raise ValueError("parent must be [N, K_in] and children [N, M, K_out]")
    return parent.transpose(0, 1) @ children.mean(dim=1) / parent.shape[0]


def estimate_edge_operators(batch: ChainFeatureBatch) -> List[Tensor]:
    """Adjacent operators C_{l,l+1} from an ENCODED chain batch."""

    return [
        conditional_cross_operator(batch.chain[edge], batch.children[edge])
        for edge in range(batch.num_levels - 1)
    ]


def estimate_endpoint_operator(batch: ChainFeatureBatch) -> Tensor:
    """Endpoint operator C_dir from an ENCODED chain batch.

    Uses full-path descendants of the root when available (variance
    reduction), otherwise the realized leaf state.
    """

    root = batch.chain[0]
    if batch.endpoint_descendants is not None:
        leaf = batch.endpoint_descendants
    else:
        leaf = batch.chain[-1].unsqueeze(1)
    return conditional_cross_operator(root, leaf)


def split_chain_batch(
    batch: ChainFeatureBatch,
    fractions: Sequence[float],
    generator: Optional[torch.Generator] = None,
) -> List[ChainFeatureBatch]:
    """Partition the parent axis into disjoint sub-batches.

    Parents are the exchangeable unit, so the split randomizes parent
    indices.  Fractions must sum to one; the last part absorbs rounding.
    """

    if len(fractions) < 2:
        raise ValueError("at least two split fractions are required")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to one")
    parents = batch.num_parents
    permutation = torch.randperm(parents, generator=generator)
    counts = [int(round(fraction * parents)) for fraction in fractions[:-1]]
    counts.append(parents - sum(counts))
    if min(counts) < 2:
        raise ValueError("every split part needs at least two parents")
    parts: List[ChainFeatureBatch] = []
    offset = 0
    for count in counts:
        indices = permutation[offset:offset + count]
        offset += count
        parts.append(_index_batch(batch, indices))
    return parts


def _index_batch(batch: ChainFeatureBatch, indices: Tensor) -> ChainFeatureBatch:
    return ChainFeatureBatch(
        chain=[states[indices] for states in batch.chain],
        children=[descendants[indices] for descendants in batch.children],
        endpoint_descendants=(
            batch.endpoint_descendants[indices]
            if batch.endpoint_descendants is not None
            else None
        ),
    )


def crossfit_edge_and_endpoint(
    batch: ChainFeatureBatch,
    generator: Optional[torch.Generator] = None,
) -> List[Tuple[List[Tensor], Tensor]]:
    """Two-fold cross-fit: edges from one half, endpoint from the other,
    then the roles swap.  Returns [(edges, c_dir), (edges, c_dir)].

    The batch must already be encoded in frozen Stage-B coordinates fitted
    on data disjoint from this batch.
    """

    first, second = split_chain_batch(batch, [0.5, 0.5], generator=generator)
    return [
        (estimate_edge_operators(first), estimate_endpoint_operator(second)),
        (estimate_edge_operators(second), estimate_endpoint_operator(first)),
    ]
