"""Faithful, explicitly scoped reimplementation of CVPR 2022 HAI.

HAI is an augmentation/stage wrapper rather than a new contrastive objective.
This module instantiates the paper's SimSiam + HAI variant: four add-one view
pairs supervise four ResNet stages, every stage has an independent three-layer
projector and predictor, and the four symmetric stop-gradient losses are summed.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from .resnet import CIFARResNet, resnet18_cifar


class HAIBackbone(nn.Module):
    """CIFAR ResNet exposing the four residual stages used by HAI."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.network: CIFARResNet = resnet18_cifar(width)
        self.output_dim = self.network.output_dim
        self.stage_channels = (width, width * 2, width * 4, width * 8)

    def forward_to_stage(self, inputs: Tensor, stage: int) -> Tensor:
        if stage not in {0, 1, 2, 3}:
            raise ValueError("HAI stage must be in 0..3")
        values = self.network.stem(inputs)
        for index, layer in enumerate(
            (self.network.layer1, self.network.layer2, self.network.layer3, self.network.layer4)
        ):
            values = layer(values)
            if index == stage:
                return values
        raise RuntimeError("unreachable HAI stage")

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class DownsampleHead(nn.Module):
    """Map a shallow stage map to the final-stage shape.

    The paper specifies 3/2/1 extra convolutional layers but does not publish
    their kernel, channel, or stride choices.  For 32x32 CIFAR inputs, each
    3x3 stride-2 Conv-BN-ReLU block doubles channels and halves resolution.
    This is a documented harness adaptation, not a claimed paper setting.
    """

    def __init__(self, input_channels: int, output_channels: int, layers: int) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        channels = input_channels
        for index in range(layers):
            next_channels = output_channels if index == layers - 1 else min(output_channels, channels * 2)
            modules.extend(
                [
                    nn.Conv2d(channels, next_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = next_channels
        self.network = nn.Sequential(*modules) if modules else nn.Identity()

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class ThreeLayerProjector(nn.Module):
    """Three fully connected layers, matching the projector depth in HAI."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim, affine=False),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class HAISimSiamHeads(nn.Module):
    """Stage adapters, augmentation embedding, projectors, and predictors."""

    def __init__(
        self,
        stage_channels: Sequence[int],
        feature_dim: int,
        augmentation_dim: int,
        projection_hidden_dim: int,
        projection_dim: int,
        predictor_hidden_dim: int,
    ) -> None:
        super().__init__()
        if len(stage_channels) != 4:
            raise ValueError("HAI requires exactly four backbone stages")
        self.adapters = nn.ModuleList(
            [
                DownsampleHead(stage_channels[0], feature_dim, 3),
                DownsampleHead(stage_channels[1], feature_dim, 2),
                DownsampleHead(stage_channels[2], feature_dim, 1),
                DownsampleHead(stage_channels[3], feature_dim, 0),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # HAI embeds the actual [brightness, contrast, saturation, hue]
        # parameters with Linear-BN-ReLU and concatenates them before h_i.
        self.augmentation_embedding = nn.Sequential(
            nn.Linear(4, augmentation_dim, bias=False),
            nn.BatchNorm1d(augmentation_dim),
            nn.ReLU(inplace=True),
        )
        expanded_dim = feature_dim + augmentation_dim
        self.projectors = nn.ModuleList(
            [ThreeLayerProjector(expanded_dim, projection_hidden_dim, projection_dim) for _ in range(4)]
        )
        self.predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(projection_dim, predictor_hidden_dim, bias=False),
                    nn.BatchNorm1d(predictor_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(predictor_hidden_dim, projection_dim),
                )
                for _ in range(4)
            ]
        )

    @staticmethod
    def negative_cosine(prediction: Tensor, target: Tensor) -> Tensor:
        return -functional.cosine_similarity(prediction, target.detach(), dim=1).mean()

    def stage_projection(self, stage_map: Tensor, parameters: Tensor, stage: int) -> Tensor:
        features = self.pool(self.adapters[stage](stage_map)).flatten(1)
        augmentation = self.augmentation_embedding(parameters)
        return self.projectors[stage](torch.cat((features, augmentation), dim=1))

    def diagnostic_projection(self, features: Tensor) -> Tensor:
        """Final-stage projector diagnostic using neutral color parameters."""
        parameters = features.new_tensor([1.0, 1.0, 1.0, 0.0]).expand(len(features), -1)
        augmentation = self.augmentation_embedding(parameters)
        return self.projectors[3](torch.cat((features, augmentation), dim=1))

    def loss(self, backbone: HAIBackbone, views: Tensor, parameters: Tensor) -> tuple[Tensor, list[Tensor]]:
        if views.ndim != 5 or views.shape[1] != 8:
            raise ValueError("HAI expects views with shape [batch, 8, channels, height, width]")
        if parameters.shape != (views.shape[0], 8, 4):
            raise ValueError("HAI expects augmentation parameters with shape [batch, 8, 4]")
        stage_losses: list[Tensor] = []
        for stage in range(4):
            left_index = 2 * stage
            pair_images = torch.cat((views[:, left_index], views[:, left_index + 1]), dim=0)
            pair_parameters = torch.cat(
                (parameters[:, left_index], parameters[:, left_index + 1]), dim=0
            )
            stage_map = backbone.forward_to_stage(pair_images, stage)
            projection = self.stage_projection(stage_map, pair_parameters, stage)
            left, right = projection.chunk(2, dim=0)
            stage_loss = 0.5 * (
                self.negative_cosine(self.predictors[stage](left), right)
                + self.negative_cosine(self.predictors[stage](right), left)
            )
            stage_losses.append(stage_loss)
        # The paper's L_overall is a sum, not a mean over stage losses.
        return torch.stack(stage_losses).sum(), stage_losses
