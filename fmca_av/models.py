"""Neural feature maps and parent-view aggregation modules."""

from typing import Iterable, List

import torch
from torch import Tensor, nn


def _activation(name: str) -> nn.Module:
    choices = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
    if name not in choices:
        raise ValueError(f"unsupported activation {name!r}")
    return choices[name]()


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int] = (128, 128),
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        dimensions: List[int] = [input_dim, *list(hidden_dims), output_dim]
        layers: List[nn.Module] = []
        for index, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(dimensions) - 2:
                layers.append(_activation(activation))
        self.network = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class ParentAggregator(nn.Module):
    """Convert a set of view embeddings into the parent-side feature input."""

    def __init__(
        self,
        feature_dim: int,
        mode: str = "mean",
        hidden_dim: int = 256,
        num_views: int = 1,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.num_views = num_views
        self.output_dim = feature_dim * num_views if mode == "concat" else feature_dim
        if mode == "deepsets":
            self.phi = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU())
            self.rho = nn.Linear(hidden_dim, feature_dim)
        elif mode in {"mean", "first", "raw", "concat"}:
            self.phi = nn.Identity()
            self.rho = nn.Identity()
        else:
            raise ValueError("parent aggregation must be one of: mean, first, raw, concat, deepsets")

    def forward(self, view_features: Tensor) -> Tensor:
        if view_features.ndim != 3:
            raise ValueError("view_features must have shape [batch, views, features]")
        if self.mode in {"first", "raw"}:
            return view_features[:, 0]
        if self.mode == "concat":
            if view_features.shape[1] != self.num_views:
                raise ValueError("concat aggregation received a different number of views than configured")
            return view_features.flatten(1)
        if self.mode == "mean":
            return view_features.mean(dim=1)
        return self.rho(self.phi(view_features).mean(dim=1))
