"""The six frozen analytic cases (Wave 0 acceptance battery).

Each case fixes a Markov chain X0 -> X1 -> X2 by construction, a feature
process z_l that is population-centered and population-white, the exact
population edge and endpoint operators, and a nested-chain sampler that
produces conditionally independent descendants of each REALIZED parent
(chain state l+1 is descendant 0 of chain state l, matching the view-tree
convention).

Frozen table (2026-08-15):
  1 nilpotent interface : per-edge Frobenius^2 = 1 each, product = 0, direct = 0
  2 hallucinated path   : C01 = C12 = 1/sqrt(2), comp = 1/2, direct = 0
  3 leaky interface     : comp = 0, direct = 1
  4 zero operator       : comp = direct = 0 (perfect closure, zero transmission)
  5 iso-spectral pair   : spectra equal, directions orthogonal, alignment = 0
  6 closed chain        : comp = direct = diag(prod rho), the only certifying case
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch import Tensor

from .estimation import ChainFeatureBatch
from .triplet import CertificateReport, certificate_report

_DTYPE = torch.float64


def _gaussian(*shape: int, generator: Optional[torch.Generator] = None) -> Tensor:
    return torch.randn(*shape, generator=generator, dtype=_DTYPE)


def _rademacher(*shape: int, generator: Optional[torch.Generator] = None) -> Tensor:
    return torch.randint(0, 2, shape, generator=generator, dtype=torch.int64).to(_DTYPE) * 2.0 - 1.0


Sampler = Callable[[int, int, int, Optional[torch.Generator]], ChainFeatureBatch]


@dataclass
class Counterexample:
    name: str
    description: str
    population_edges: List[Tensor]
    population_direct: Tensor
    certifies: bool
    sampler: Sampler

    def population_report(self, top_k: int = 4) -> CertificateReport:
        return certificate_report(self.population_direct, edges=self.population_edges, top_k=top_k)

    def sample(
        self,
        parents: int,
        children_per_edge: int = 4,
        endpoint_descendants: int = 4,
        generator: Optional[torch.Generator] = None,
    ) -> ChainFeatureBatch:
        return self.sampler(parents, children_per_edge, endpoint_descendants, generator)


def case_nilpotent() -> Counterexample:
    """z0=(A,B), z1=(A,C), z2=(D,C): both edges carry unit Frobenius mass but
    the ordered product (and the true endpoint operator) is zero."""

    edge_01 = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=_DTYPE)
    edge_12 = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=_DTYPE)
    direct = torch.zeros(2, 2, dtype=_DTYPE)

    def sampler(parents: int, m: int, endpoint_m: int, generator: Optional[torch.Generator]) -> ChainFeatureBatch:
        a, b = _gaussian(parents, generator=generator), _gaussian(parents, generator=generator)
        z0 = torch.stack([a, b], dim=1)
        c_children = _gaussian(parents, m, generator=generator)
        children_0 = torch.stack([a.unsqueeze(1).expand(-1, m), c_children], dim=2)
        c_state = c_children[:, 0]
        d_children = _gaussian(parents, m, generator=generator)
        children_1 = torch.stack([d_children, c_state.unsqueeze(1).expand(-1, m)], dim=2)
        endpoint = torch.stack(
            [
                _gaussian(parents, endpoint_m, generator=generator),
                _gaussian(parents, endpoint_m, generator=generator),
            ],
            dim=2,
        )
        return ChainFeatureBatch(
            chain=[z0, children_0[:, 0], children_1[:, 0]],
            children=[children_0, children_1],
            endpoint_descendants=endpoint,
        )

    return Counterexample(
        name="nilpotent_interface",
        description="strong local edges, zero ordered product, zero endpoint",
        population_edges=[edge_01, edge_12],
        population_direct=direct,
        certifies=False,
        sampler=sampler,
    )


def case_hallucinated_path() -> Counterexample:
    """A,B independent Rademacher; z0=A, z1=(A+B)/sqrt(2), z2=B: the ledger
    claims 1/2 while the true endpoint dependence is zero."""

    root_half = 1.0 / math.sqrt(2.0)
    edge_01 = torch.tensor([[root_half]], dtype=_DTYPE)
    edge_12 = torch.tensor([[root_half]], dtype=_DTYPE)
    direct = torch.zeros(1, 1, dtype=_DTYPE)

    def sampler(parents: int, m: int, endpoint_m: int, generator: Optional[torch.Generator]) -> ChainFeatureBatch:
        a = _rademacher(parents, generator=generator)
        b_children = _rademacher(parents, m, generator=generator)
        z1_children = ((a.unsqueeze(1) + b_children) * root_half).unsqueeze(2)
        b_state = b_children[:, 0]
        # X2 = B is a deterministic function of the realized X1 = (A, B).
        z2_children = b_state.reshape(parents, 1, 1).expand(-1, m, -1).clone()
        endpoint = _rademacher(parents, endpoint_m, generator=generator).unsqueeze(2)
        return ChainFeatureBatch(
            chain=[a.unsqueeze(1), z1_children[:, 0], z2_children[:, 0]],
            children=[z1_children, z2_children],
            endpoint_descendants=endpoint,
        )

    return Counterexample(
        name="hallucinated_path",
        description="interface mixes independent bits; ledger fabricates dependence",
        population_edges=[edge_01, edge_12],
        population_direct=direct,
        certifies=False,
        sampler=sampler,
    )


def case_leaky_interface() -> Counterexample:
    """z0=A, z1=C, z2=A with A independent of C: the endpoint dependence is
    perfect but the interface transmits none of it."""

    edge_01 = torch.zeros(1, 1, dtype=_DTYPE)
    edge_12 = torch.zeros(1, 1, dtype=_DTYPE)
    direct = torch.tensor([[1.0]], dtype=_DTYPE)

    def sampler(parents: int, m: int, endpoint_m: int, generator: Optional[torch.Generator]) -> ChainFeatureBatch:
        a = _rademacher(parents, generator=generator)
        c_children = _rademacher(parents, m, generator=generator).unsqueeze(2)
        # X2 = A is a deterministic function of the realized X1 = (A, C).
        z2_children = a.reshape(parents, 1, 1).expand(-1, m, -1).clone()
        endpoint = a.reshape(parents, 1, 1).expand(-1, endpoint_m, -1).clone()
        return ChainFeatureBatch(
            chain=[a.unsqueeze(1), c_children[:, 0], z2_children[:, 0]],
            children=[c_children, z2_children],
            endpoint_descendants=endpoint,
        )

    return Counterexample(
        name="leaky_interface",
        description="perfect endpoint dependence invisible to the interface ledger",
        population_edges=[edge_01, edge_12],
        population_direct=direct,
        certifies=False,
        sampler=sampler,
    )


def case_zero_operator() -> Counterexample:
    """Everything independent: closure is perfect and transmission is zero,
    which is why the training loss must keep the endpoint dependence term."""

    zero = torch.zeros(1, 1, dtype=_DTYPE)

    def sampler(parents: int, m: int, endpoint_m: int, generator: Optional[torch.Generator]) -> ChainFeatureBatch:
        z0 = _gaussian(parents, 1, generator=generator)
        children_0 = _gaussian(parents, m, 1, generator=generator)
        children_1 = _gaussian(parents, m, 1, generator=generator)
        endpoint = _gaussian(parents, endpoint_m, 1, generator=generator)
        return ChainFeatureBatch(
            chain=[z0, children_0[:, 0], children_1[:, 0]],
            children=[children_0, children_1],
            endpoint_descendants=endpoint,
        )

    return Counterexample(
        name="zero_operator",
        description="perfect closure with zero transmission",
        population_edges=[zero.clone(), zero.clone()],
        population_direct=zero.clone(),
        certifies=False,
        sampler=sampler,
    )


def case_isospectral_mismatch() -> Counterexample:
    """C_dir = diag(1/2, 0) and C_comp = diag-complement with the SAME
    spectrum {1/2, 0} but orthogonal directions: comparing singular values
    alone sees nothing, the full-matrix residual sees everything.

    Construction: A,B,C,E,F independent standard normals.
      X0 = (A, B); X1 keeps (A, B) and draws fresh (C, E); X2 draws fresh F.
      z0 = (A, B); z1 = ((B + C)/sqrt(2), E); z2 = (A/2 + sqrt(3)/2 F, C).
    """

    root_half = 1.0 / math.sqrt(2.0)
    edge_01 = torch.tensor([[0.0, 0.0], [root_half, 0.0]], dtype=_DTYPE)
    edge_12 = torch.tensor([[0.0, root_half], [0.0, 0.0]], dtype=_DTYPE)
    direct = torch.tensor([[0.5, 0.0], [0.0, 0.0]], dtype=_DTYPE)
    scale_a = 0.5
    scale_f = math.sqrt(3.0) / 2.0

    def sampler(parents: int, m: int, endpoint_m: int, generator: Optional[torch.Generator]) -> ChainFeatureBatch:
        a, b = _gaussian(parents, generator=generator), _gaussian(parents, generator=generator)
        z0 = torch.stack([a, b], dim=1)
        c_children = _gaussian(parents, m, generator=generator)
        e_children = _gaussian(parents, m, generator=generator)
        children_0 = torch.stack(
            [(b.unsqueeze(1) + c_children) * root_half, e_children], dim=2
        )
        c_state = c_children[:, 0]
        f_children = _gaussian(parents, m, generator=generator)
        children_1 = torch.stack(
            [
                scale_a * a.unsqueeze(1) + scale_f * f_children,
                c_state.unsqueeze(1).expand(-1, m),
            ],
            dim=2,
        )
        c_fresh = _gaussian(parents, endpoint_m, generator=generator)
        f_fresh = _gaussian(parents, endpoint_m, generator=generator)
        endpoint = torch.stack(
            [scale_a * a.unsqueeze(1) + scale_f * f_fresh, c_fresh], dim=2
        )
        return ChainFeatureBatch(
            chain=[z0, children_0[:, 0], children_1[:, 0]],
            children=[children_0, children_1],
            endpoint_descendants=endpoint,
        )

    return Counterexample(
        name="isospectral_mismatch",
        description="identical spectra, orthogonal directions; full-matrix residual required",
        population_edges=[edge_01, edge_12],
        population_direct=direct,
        certifies=False,
        sampler=sampler,
    )


def case_closed_chain(
    rho_first: Optional[Tensor] = None,
    rho_second: Optional[Tensor] = None,
) -> Counterexample:
    """Positive case: coordinatewise AR channels with lossless interfaces.
    The ordered product equals the endpoint operator exactly and the
    certificate must certify the nonzero transmitted modes."""

    rho_0 = rho_first if rho_first is not None else torch.tensor([0.9, 0.7, 0.5], dtype=_DTYPE)
    rho_1 = rho_second if rho_second is not None else torch.tensor([0.8, 0.6, 0.4], dtype=_DTYPE)
    if rho_0.shape != rho_1.shape or rho_0.ndim != 1:
        raise ValueError("rho vectors must be one-dimensional with equal length")
    edge_01 = torch.diag(rho_0)
    edge_12 = torch.diag(rho_1)
    direct = torch.diag(rho_0 * rho_1)

    def _children(state: Tensor, rho: Tensor, m: int, generator: Optional[torch.Generator]) -> Tensor:
        noise = _gaussian(state.shape[0], m, state.shape[1], generator=generator)
        return rho * state.unsqueeze(1) + (1.0 - rho.square()).sqrt() * noise

    def sampler(parents: int, m: int, endpoint_m: int, generator: Optional[torch.Generator]) -> ChainFeatureBatch:
        z0 = _gaussian(parents, rho_0.shape[0], generator=generator)
        children_0 = _children(z0, rho_0, m, generator)
        z1 = children_0[:, 0]
        children_1 = _children(z1, rho_1, m, generator)
        z1_fresh = _children(z0, rho_0, endpoint_m, generator)
        endpoint = torch.stack(
            [
                _children(z1_fresh[:, index], rho_1, 1, generator)[:, 0]
                for index in range(endpoint_m)
            ],
            dim=1,
        )
        return ChainFeatureBatch(
            chain=[z0, z1, children_1[:, 0]],
            children=[children_0, children_1],
            endpoint_descendants=endpoint,
        )

    return Counterexample(
        name="closed_chain",
        description="true closure with nonzero transmission; the only certifying case",
        population_edges=[edge_01, edge_12],
        population_direct=direct,
        certifies=True,
        sampler=sampler,
    )


def all_counterexamples() -> List[Counterexample]:
    return [
        case_nilpotent(),
        case_hallucinated_path(),
        case_leaky_interface(),
        case_zero_operator(),
        case_isospectral_mismatch(),
        case_closed_chain(),
    ]
