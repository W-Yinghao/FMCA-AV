"""Residual-block-resolution feature walks for pretrained backbones.

The probing-depth study measures certificates at every block interface
of a pretrained model (torchvision ResNets or this repo's CIFAR
ResNet).  Stochasticity comes from the augmentation channel: the walk
itself is deterministic per image, and the caller pairs walks of
independent views to form well-defined operators.
"""

from typing import List

import torch
from torch import Tensor, nn

from ..resnet import CIFARResNet


@torch.no_grad()
def blockwise_pooled_features(model: nn.Module, images: Tensor) -> List[Tensor]:
    """Pooled features after the stem and after EVERY residual block."""

    if isinstance(model, CIFARResNet):
        values = model.stem(images)
        stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    else:
        # torchvision ResNet family
        values = model.conv1(images)
        values = model.bn1(values)
        values = model.relu(values)
        values = model.maxpool(values)
        stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    features: List[Tensor] = [values.mean(dim=(-2, -1))]
    for stage in stages:
        for block in stage:
            values = block(values)
            features.append(values.mean(dim=(-2, -1)))
    return features


def block_count(model: nn.Module) -> int:
    if isinstance(model, CIFARResNet):
        stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    else:
        stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    return 1 + sum(len(stage) for stage in stages)
