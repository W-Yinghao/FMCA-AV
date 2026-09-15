"""AR(1)-Hermite analytic reference chain.

For a stationary Gaussian AR(1) chain X_{l+1} = rho_l X_l + sqrt(1-rho_l^2) eps
with standard normal marginals, the normalized probabilists' Hermite
polynomials he_k(x) = He_k(x)/sqrt(k!) are exact conditional-expectation
eigenfunctions:  E[he_k(X_{l+1}) | X_l = x] = rho_l^k he_k(x).

With feature map z_l = (he_k(X_l))_{k in orders_l}, every population edge
operator is an exact (rectangular) diagonal selection

    C_{l,l+1}[i, j] = rho_l^k   if orders_l[i] == orders_{l+1}[j] == k else 0,

the endpoint operator uses rho_total = prod_l rho_l, and the ordered product
keeps mode k iff k survives EVERY interior interface.  Dropping Hermite
orders at an interior level therefore produces a closed-form projection
defect Delta_CK (Theorem 1 telescoping made analytic), with no Nystrom step.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import Tensor

from .estimation import ChainFeatureBatch

_DTYPE = torch.float64


def normalized_hermite_features(samples: Tensor, orders: Sequence[int]) -> Tensor:
    """he_k(x) = He_k(x)/sqrt(k!) for the requested orders (k >= 1)."""

    if samples.ndim != 1:
        raise ValueError("samples must be a one-dimensional tensor of chain states")
    if len(orders) < 1 or min(orders) < 1:
        raise ValueError("orders must be nonempty positive Hermite degrees")
    highest = max(orders)
    polynomials = [torch.ones_like(samples), samples]
    for degree in range(1, highest):
        polynomials.append(samples * polynomials[degree] - degree * polynomials[degree - 1])
    columns = [polynomials[k] / math.sqrt(math.factorial(k)) for k in orders]
    return torch.stack(columns, dim=1)


def population_edge_operator(
    rho: float,
    orders_in: Sequence[int],
    orders_out: Sequence[int],
) -> Tensor:
    operator = torch.zeros(len(orders_in), len(orders_out), dtype=_DTYPE)
    for row, order_in in enumerate(orders_in):
        for column, order_out in enumerate(orders_out):
            if order_in == order_out:
                operator[row, column] = rho ** order_in
    return operator


@dataclass
class GaussianHermiteChain:
    """A depth-L AR chain with per-level Hermite interfaces.

    rhos:         one correlation per edge (length L)
    level_orders: Hermite orders kept by the interface at each level
                  (length L + 1); interior truncation creates the defect.
    """

    rhos: List[float]
    level_orders: List[List[int]]

    def __post_init__(self) -> None:
        if len(self.level_orders) != len(self.rhos) + 1:
            raise ValueError("need one order list per level (edges + 1)")
        for level, orders in enumerate(self.level_orders):
            if len(set(orders)) != len(orders):
                raise ValueError(f"level {level} repeats Hermite orders: {orders}")
        for rho in self.rhos:
            if not -1.0 < rho < 1.0:
                raise ValueError("every rho must lie strictly inside (-1, 1)")

    @property
    def num_edges(self) -> int:
        return len(self.rhos)

    def population_edges(self) -> List[Tensor]:
        return [
            population_edge_operator(rho, self.level_orders[edge], self.level_orders[edge + 1])
            for edge, rho in enumerate(self.rhos)
        ]

    def population_direct(self) -> Tensor:
        rho_total = math.prod(self.rhos)
        return population_edge_operator(rho_total, self.level_orders[0], self.level_orders[-1])

    def population_composed(self) -> Tensor:
        """Closed form: mode k survives iff every interior level keeps it."""

        surviving = set(self.level_orders[0])
        for interior in self.level_orders[1:-1]:
            surviving &= set(interior)
        rho_total = math.prod(self.rhos)
        operator = torch.zeros(
            len(self.level_orders[0]), len(self.level_orders[-1]), dtype=_DTYPE
        )
        for row, order_in in enumerate(self.level_orders[0]):
            for column, order_out in enumerate(self.level_orders[-1]):
                if order_in == order_out and order_in in surviving:
                    operator[row, column] = rho_total ** order_in
        return operator

    def population_defect(self) -> Tensor:
        return self.population_direct() - self.population_composed()

    def _step(self, states: Tensor, rho: float, count: int, generator: Optional[torch.Generator]) -> Tensor:
        noise = torch.randn(states.shape[0], count, generator=generator, dtype=_DTYPE)
        return rho * states.unsqueeze(1) + math.sqrt(1.0 - rho * rho) * noise

    def sample(
        self,
        parents: int,
        children_per_edge: int = 4,
        endpoint_descendants: int = 4,
        generator: Optional[torch.Generator] = None,
    ) -> ChainFeatureBatch:
        """Nested-chain features; chain state l+1 is descendant 0 of state l."""

        states = [torch.randn(parents, generator=generator, dtype=_DTYPE)]
        children_states: List[Tensor] = []
        for edge, rho in enumerate(self.rhos):
            descendants = self._step(states[edge], rho, children_per_edge, generator)
            children_states.append(descendants)
            states.append(descendants[:, 0])
        endpoint_states = states[0].unsqueeze(1).expand(-1, endpoint_descendants).clone()
        for rho in self.rhos:
            noise = torch.randn(
                parents, endpoint_descendants, generator=generator, dtype=_DTYPE
            )
            endpoint_states = rho * endpoint_states + math.sqrt(1.0 - rho * rho) * noise
        chain = [
            normalized_hermite_features(states[level], self.level_orders[level])
            for level in range(self.num_edges + 1)
        ]
        children = [
            torch.stack(
                [
                    normalized_hermite_features(
                        children_states[edge][:, view], self.level_orders[edge + 1]
                    )
                    for view in range(children_per_edge)
                ],
                dim=1,
            )
            for edge in range(self.num_edges)
        ]
        endpoint = torch.stack(
            [
                normalized_hermite_features(endpoint_states[:, view], self.level_orders[-1])
                for view in range(endpoint_descendants)
            ],
            dim=1,
        )
        return ChainFeatureBatch(chain=chain, children=children, endpoint_descendants=endpoint)
