"""Stage-tapped CIFAR ResNet exposing per-stage pooled representations.

Levels of the view hierarchy attach to backbone stages through a
level -> stage map; level l's projector consumes the pooled output of its
assigned stage.  Running a level-l view only up to its stage keeps the
depth-aligned semantics (one view refinement per backbone stage step).
"""

from typing import List, Optional

from torch import Tensor, nn

from ..resnet import resnet18_cifar


class StageTappedCIFARResNet(nn.Module):
    def __init__(self, width: int = 64, stem: str = "cifar") -> None:
        super().__init__()
        resnet = resnet18_cifar(width=width)
        if stem not in {"cifar", "downsample"}:
            raise ValueError('stem must be "cifar" or "downsample"')
        self.stem = resnet.stem
        if stem == "downsample":
            # 64-pixel inputs at CIFAR cost: halve twice up front so the
            # stage resolutions match what the CIFAR runs saw.
            self.stem = nn.Sequential(
                nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=1, padding=1),
            )
        self.stages = nn.ModuleList([resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4])
        self.pool = resnet.pool
        self.stage_dims = [width, width * 2, width * 4, width * 8]
        self.output_dim = resnet.output_dim

    def forward_stages(self, inputs: Tensor, up_to: Optional[int] = None) -> List[Tensor]:
        """Pooled representations after each stage, truncated at ``up_to``."""

        last = len(self.stages) - 1 if up_to is None else int(up_to)
        if not 0 <= last < len(self.stages):
            raise ValueError(f"up_to must index one of {len(self.stages)} stages")
        values = self.stem(inputs)
        pooled: List[Tensor] = []
        for index in range(last + 1):
            values = self.stages[index](values)
            pooled.append(self.pool(values).flatten(1))
        return pooled

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_stages(inputs)[-1]
