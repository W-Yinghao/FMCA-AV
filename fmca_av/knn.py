"""Weighted frozen-backbone k-nearest-neighbor evaluation."""

from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


@torch.inference_mode()
def feature_bank(
    backbone: nn.Module,
    loader: Iterable[Tuple[Tensor, Tensor]],
    device: torch.device,
    storage_device: Optional[torch.device] = None,
    storage_dtype: torch.dtype = torch.float32,
    limit: int = 0,
) -> Tuple[Tensor, Tensor]:
    features = []
    labels = []
    collected = 0
    backbone.eval()
    for images, batch_labels in loader:
        encoded = functional.normalize(backbone(images.to(device)), dim=1)
        destination = storage_device or device
        features.append(encoded.to(device=destination, dtype=storage_dtype))
        labels.append(batch_labels.to(destination))
        collected += encoded.shape[0]
        if limit and collected >= limit:
            break
    bank = torch.cat(features)
    bank_labels = torch.cat(labels)
    if limit:
        bank = bank[:limit]
        bank_labels = bank_labels[:limit]
    return bank, bank_labels


@torch.inference_mode()
def weighted_knn_accuracy(
    backbone: nn.Module,
    train_loader: Iterable[Tuple[Tensor, Tensor]],
    test_loader: Iterable[Tuple[Tensor, Tensor]],
    classes: int,
    device: torch.device,
    neighbors: int = 20,
    temperature: float = 0.07,
) -> float:
    bank, bank_labels = feature_bank(backbone, train_loader, device)
    if neighbors > bank.shape[0]:
        raise ValueError("neighbors exceeds the feature-bank size")
    correct = 0
    total = 0
    for images, labels in test_loader:
        query = functional.normalize(backbone(images.to(device)), dim=1)
        similarities, indices = (query @ bank.transpose(0, 1)).topk(neighbors, dim=1)
        neighbor_labels = bank_labels[indices]
        weights = torch.exp(similarities / temperature)
        votes = torch.zeros(query.shape[0], classes, device=device)
        votes.scatter_add_(1, neighbor_labels, weights)
        predictions = votes.argmax(dim=1).cpu()
        correct += int((predictions == labels).sum())
        total += labels.numel()
    return correct / total


@torch.inference_mode()
def weighted_knn_accuracy_chunked(
    backbone: nn.Module,
    train_loader: Iterable[Tuple[Tensor, Tensor]],
    test_loader: Iterable[Tuple[Tensor, Tensor]],
    classes: int,
    device: torch.device,
    neighbors: int = 20,
    temperature: float = 0.07,
    bank_chunk_size: int = 8192,
    bank_limit: int = 0,
) -> Tuple[float, int]:
    """Exact chunked search over a CPU feature bank, optionally capped for profiling."""
    bank, bank_labels = feature_bank(
        backbone,
        train_loader,
        device,
        storage_device=torch.device("cpu"),
        storage_dtype=torch.float16,
        limit=bank_limit,
    )
    if neighbors > bank.shape[0]:
        raise ValueError("neighbors exceeds the feature-bank size")
    correct = 0
    total = 0
    backbone.eval()
    for images, labels in test_loader:
        query = functional.normalize(backbone(images.to(device)), dim=1).float()
        best_scores = torch.empty(query.shape[0], 0, device=device)
        best_labels = torch.empty(query.shape[0], 0, dtype=torch.long, device=device)
        for start in range(0, bank.shape[0], bank_chunk_size):
            stop = min(start + bank_chunk_size, bank.shape[0])
            chunk = bank[start:stop].to(device=device, dtype=query.dtype)
            similarities = query @ chunk.transpose(0, 1)
            count = min(neighbors, similarities.shape[1])
            chunk_scores, chunk_indices = similarities.topk(count, dim=1)
            chunk_labels = bank_labels[start:stop].to(device)[chunk_indices]
            candidate_scores = torch.cat((best_scores, chunk_scores), dim=1)
            candidate_labels = torch.cat((best_labels, chunk_labels), dim=1)
            keep = min(neighbors, candidate_scores.shape[1])
            best_scores, positions = candidate_scores.topk(keep, dim=1)
            best_labels = candidate_labels.gather(1, positions)
        weights = torch.exp(best_scores / temperature)
        votes = torch.zeros(query.shape[0], classes, device=device)
        votes.scatter_add_(1, best_labels, weights)
        predictions = votes.argmax(dim=1).cpu()
        correct += int((predictions == labels).sum())
        total += labels.numel()
    return correct / total, int(bank.shape[0])
