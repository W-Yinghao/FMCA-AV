"""Small ResNet backbones without a torchvision runtime dependency."""

from typing import List, Type

from torch import Tensor, nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        if stride != 1 or input_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.shortcut(inputs)
        values = self.relu(self.bn1(self.conv1(inputs)))
        values = self.bn2(self.conv2(values))
        return self.relu(values + residual)


class CIFARResNet(nn.Module):
    def __init__(self, block: Type[BasicBlock], blocks: List[int], width: int = 64) -> None:
        super().__init__()
        self.channels = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._layer(block, width, blocks[0], stride=1)
        self.layer2 = self._layer(block, width * 2, blocks[1], stride=2)
        self.layer3 = self._layer(block, width * 4, blocks[2], stride=2)
        self.layer4 = self._layer(block, width * 8, blocks[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = width * 8 * block.expansion
        self._initialize()

    def _layer(self, block: Type[BasicBlock], channels: int, count: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (count - 1)
        layers = []
        for item_stride in strides:
            layers.append(block(self.channels, channels, item_stride))
            self.channels = channels * block.expansion
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        values = self.stem(inputs)
        values = self.layer1(values)
        values = self.layer2(values)
        values = self.layer3(values)
        values = self.layer4(values)
        return self.pool(values).flatten(1)


def resnet18_cifar(width: int = 64) -> CIFARResNet:
    return CIFARResNet(BasicBlock, [2, 2, 2, 2], width=width)

