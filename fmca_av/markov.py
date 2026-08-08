"""Exact finite-state Markov operator diagnostics."""

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import Tensor


@dataclass
class MarkovSpectrum:
    stationary: Tensor
    singular_values: Tensor


def stationary_distribution(transition: Tensor) -> Tensor:
    values, vectors = torch.linalg.eig(transition.transpose(0, 1))
    index = int(torch.argmin((values - 1).abs()))
    vector = vectors[:, index].real
    if vector.sum() < 0:
        vector = -vector
    vector = vector.clamp_min(0)
    return vector / vector.sum()


def lag_spectrum(transition: Tensor, lag: int) -> MarkovSpectrum:
    transition = transition.double()
    stationary = stationary_distribution(transition)
    lagged = torch.linalg.matrix_power(transition, lag)
    root = stationary.sqrt()
    normalized = root[:, None] * lagged / root[None, :]
    singular = torch.linalg.svdvals(normalized)
    # The first singular value is the constant stationary mode.
    return MarkovSpectrum(stationary=stationary, singular_values=singular[1:])


def reversible_chain(states: int, generator: torch.Generator) -> Tensor:
    weights = torch.rand(states, states, generator=generator, dtype=torch.float64)
    weights = (weights + weights.transpose(0, 1)) / 2
    weights += torch.eye(states, dtype=torch.float64) * states
    return weights / weights.sum(dim=1, keepdim=True)


def directed_cycle(states: int, clockwise: float = 0.78, stay: float = 0.15) -> Tensor:
    transition = torch.zeros(states, states, dtype=torch.float64)
    counterclockwise = 1 - clockwise - stay
    for index in range(states):
        transition[index, index] = stay
        transition[index, (index + 1) % states] = clockwise
        transition[index, (index - 1) % states] = counterclockwise
    return transition


def nonnormal_chain(states: int, generator: torch.Generator) -> Tensor:
    values = torch.rand(states, states, generator=generator, dtype=torch.float64).square()
    direction = torch.arange(states)
    values[direction, (direction + 1) % states] += 5.0
    values[direction, direction] += torch.linspace(0.2, 3.0, states)
    return values / values.sum(dim=1, keepdim=True)


def metastable_chain(states: int, generator: torch.Generator) -> Tensor:
    if states % 2:
        raise ValueError("metastable chain requires an even state count")
    half = states // 2
    weights = torch.full((states, states), 0.002, dtype=torch.float64)
    weights[:half, :half] += torch.rand(half, half, generator=generator, dtype=torch.float64)
    weights[half:, half:] += torch.rand(half, half, generator=generator, dtype=torch.float64)
    weights = (weights + weights.transpose(0, 1)) / 2
    weights += torch.eye(states, dtype=torch.float64)
    return weights / weights.sum(dim=1, keepdim=True)


def lag_composition_diagnostic(transition: Tensor, lags: List[int], modes: int = 8) -> Dict[str, object]:
    one_step = lag_spectrum(transition, 1).singular_values[:modes]
    records = []
    for lag in lags:
        direct = lag_spectrum(transition, lag).singular_values[:modes]
        predicted = one_step.pow(lag)
        count = min(len(direct), len(predicted))
        error = (direct[:count] - predicted[:count]).abs()
        records.append(
            {
                "lag": lag,
                "direct_singular_values": direct.tolist(),
                "one_step_power_prediction": predicted.tolist(),
                "mae": float(error.mean()),
                "max_error": float(error.max()),
            }
        )
    stationary = lag_spectrum(transition, 1).stationary
    detailed_balance = (
        stationary[:, None] * transition - stationary[None, :] * transition.transpose(0, 1)
    ).abs().max()
    normalized = stationary.sqrt()[:, None] * transition / stationary.sqrt()[None, :]
    nonnormality = torch.linalg.matrix_norm(
        normalized @ normalized.transpose(0, 1) - normalized.transpose(0, 1) @ normalized
    )
    return {
        "states": transition.shape[0],
        "stationary": stationary.tolist(),
        "one_step_singular_values": one_step.tolist(),
        "detailed_balance_max_error": float(detailed_balance),
        "normalized_operator_nonnormality": float(nonnormality),
        "lags": records,
    }
