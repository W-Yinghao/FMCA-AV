"""Frozen training objective for the compositional certificate method.

Handoff 2026-08-15, section 5 (algebra frozen):

    L = -S(C_dir) + beta * ||C_dir - C_comp||_F^2 / (||C_dir||_F^2 + eps)
        + gamma * sum_l ||R_l - I||_F^2  + alpha-term over per-edge scores,
    with S(C) = ||C||_F^2.

Sign convention implemented here: the per-edge term enters as
``- alpha * sum_l S(C_{l,l+1})`` so that alpha > 0, beta = 0 reproduces the
HFMCA/HAI additive family (maximize each local dependence), which is the
frozen contrast design; the full method uses alpha = 0 (or tiny).  The
handoff's formula line reads "+ alpha * sum S"; the additive-control
semantics require the maximizing sign, so this choice is surfaced for
confirmation rather than silently absorbed.

The endpoint term and the whitening term are NOT optional: a zero operator
satisfies closure perfectly (counterexample 4).

Train-time estimation shares batch samples for efficiency; every REPORTED
certificate must instead come from the frozen Stage-B/C protocol
(coordinates.py / estimation.py) on held-out data.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .estimation import ChainFeatureBatch
from .triplet import compose_edge_operators


def frobenius_score(operator: Tensor) -> Tensor:
    return operator.square().sum()


def normalized_score(operator: Tensor) -> Tensor:
    """||C||_F^2 / min(K_in, K_out): average squared singular mass per mode.

    Dimension normalization keeps every training term O(1) regardless of
    feature width; the raw K^2-entry sums explode the whitening penalty at
    K = 128 and blow up SGD within a few steps (gate probe 945112).
    """

    return operator.square().sum() / min(operator.shape)


def identity_penalty(moment: Tensor) -> Tensor:
    """Per-entry mean of (R - I)^2, the dimension-normalized whitening term."""

    identity = torch.eye(moment.shape[0], dtype=moment.dtype, device=moment.device)
    return (moment - identity).square().mean()


def batch_level_statistics(batch: ChainFeatureBatch) -> Tuple[List[Tensor], List[Tensor]]:
    """One shared (mean, second moment) per level from all its batch samples.

    Level l pools the chain state with the descendants living at level l so
    that the SAME train-time coordinates serve the incoming and the outgoing
    edge, mirroring the Stage-B shared-interface rule.
    """

    means: List[Tensor] = []
    moments: List[Tensor] = []
    for level in range(batch.num_levels):
        parts = [batch.chain[level]]
        if level >= 1:
            parts.append(batch.children[level - 1].flatten(0, 1))
        pooled = torch.cat(parts, dim=0)
        mean = pooled.mean(dim=0, keepdim=True)
        centered = pooled - mean
        moments.append(centered.transpose(0, 1) @ centered / centered.shape[0])
        means.append(mean)
    return means, moments


def train_edge_operators(batch: ChainFeatureBatch, means: Sequence[Tensor]) -> List[Tensor]:
    """Differentiable per-edge cross moments in shared centered coordinates."""

    edges: List[Tensor] = []
    for edge in range(batch.num_levels - 1):
        parent = batch.chain[edge] - means[edge]
        children = batch.children[edge] - means[edge + 1].unsqueeze(0)
        edges.append(parent.transpose(0, 1) @ children.mean(dim=1) / parent.shape[0])
    return edges


def train_endpoint_operator(batch: ChainFeatureBatch, means: Sequence[Tensor]) -> Tensor:
    root = batch.chain[0] - means[0]
    if batch.endpoint_descendants is not None:
        leaf = (batch.endpoint_descendants - means[-1].unsqueeze(0)).mean(dim=1)
    else:
        leaf = batch.chain[-1] - means[-1]
    return root.transpose(0, 1) @ leaf / root.shape[0]


@dataclass
class CertificateLossTerms:
    endpoint_score: Tensor
    closure_ratio: Tensor
    whitening_penalty: Tensor
    edge_score_sum: Tensor
    total: Tensor

    def as_metrics(self) -> Dict[str, float]:
        return {
            "endpoint_score": float(self.endpoint_score.detach()),
            "closure_ratio": float(self.closure_ratio.detach()),
            "whitening_penalty": float(self.whitening_penalty.detach()),
            "edge_score_sum": float(self.edge_score_sum.detach()),
            "total": float(self.total.detach()),
        }


def certificate_training_loss(
    batch: ChainFeatureBatch,
    alpha: float = 0.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    closure_stop_grad: bool = False,
) -> CertificateLossTerms:
    """Assemble the frozen loss from one differentiable chain batch.

    closure_stop_grad detaches C_dir inside the closure ratio only (the
    composition is pulled toward the measured endpoint, never the reverse);
    it is an ablation flag, not a default.
    """

    means, moments = batch_level_statistics(batch)
    edges = train_edge_operators(batch, means)
    c_comp = compose_edge_operators(edges)
    c_dir = train_endpoint_operator(batch, means)
    if c_dir.shape != c_comp.shape:
        raise ValueError("endpoint and composed operators must share a shape")

    endpoint_score = normalized_score(c_dir)
    raw_endpoint = frobenius_score(c_dir)
    closure_target = c_dir.detach() if closure_stop_grad else c_dir
    closure_denominator = (
        raw_endpoint.detach() if closure_stop_grad else raw_endpoint
    )
    closure_ratio = frobenius_score(closure_target - c_comp) / (closure_denominator + epsilon)
    whitening_penalty = torch.stack([identity_penalty(moment) for moment in moments]).sum()
    edge_score_sum = torch.stack([normalized_score(edge) for edge in edges]).sum()

    total = (
        -endpoint_score
        + beta * closure_ratio
        + gamma * whitening_penalty
        - alpha * edge_score_sum
    )
    return CertificateLossTerms(
        endpoint_score=endpoint_score,
        closure_ratio=closure_ratio,
        whitening_penalty=whitening_penalty,
        edge_score_sum=edge_score_sum,
        total=total,
    )


def cross_pair_score(
    batch: ChainFeatureBatch,
    means: Sequence[Tensor],
    level_from: int,
    level_to: int,
) -> Tensor:
    """S of a selected cross-scale operator (AMDIM-style control rows).

    Pairs the realized chain state at ``level_from`` with the descendants
    living at ``level_to`` (which requires level_to >= 1).
    """

    if not 0 <= level_from < batch.num_levels:
        raise ValueError("level_from out of range")
    if not 1 <= level_to < batch.num_levels:
        raise ValueError("level_to must be a descendant-bearing level")
    parent = batch.chain[level_from] - means[level_from]
    children = batch.children[level_to - 1] - means[level_to].unsqueeze(0)
    operator = parent.transpose(0, 1) @ children.mean(dim=1) / parent.shape[0]
    return normalized_score(operator)
