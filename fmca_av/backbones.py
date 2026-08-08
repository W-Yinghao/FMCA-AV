"""Vision backbone adapters with a uniform output_dim attribute."""

from torch import Tensor, nn
from torchvision import models

from .resnet import resnet18_cifar


class TorchvisionBackbone(nn.Module):
    def __init__(self, network: nn.Module, output_dim: int) -> None:
        super().__init__()
        self.network = network
        self.output_dim = output_dim

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class VGGFeatureBackbone(nn.Module):
    """VGG convolutional trunk with a map-compatible global representation."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__(); network.classifier = nn.Identity(); self.network = network; self.output_dim = 512

    def forward(self, inputs: Tensor) -> Tensor:
        spatial = self.network.features(inputs)
        return spatial.mean(dim=(-2, -1))


def build_backbone(name: str, width: int = 64) -> nn.Module:
    if name == "resnet18_cifar":
        return resnet18_cifar(width=width)
    if name == "resnet18_imagenet":
        network = models.resnet18(weights=None)
        output_dim = int(network.fc.in_features)
        network.fc = nn.Identity()
        return TorchvisionBackbone(network, output_dim)
    if name == "resnet50_imagenet":
        network = models.resnet50(weights=None)
        output_dim = int(network.fc.in_features)
        network.fc = nn.Identity()
        return TorchvisionBackbone(network, output_dim)
    if name == "convnext_tiny":
        network = models.convnext_tiny(weights=None)
        output_dim = int(network.classifier[-1].in_features)
        network.classifier[-1] = nn.Identity()
        return TorchvisionBackbone(network, output_dim)
    if name == "vgg16_bn":
        return VGGFeatureBackbone(models.vgg16_bn(weights=None))
    if name == "vit_b_16":
        network = models.vit_b_16(weights=None)
        output_dim = int(network.heads.head.in_features)
        network.heads.head = nn.Identity()
        return TorchvisionBackbone(network, output_dim)
    if name == "vit_s_16":
        network = models.VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
            num_classes=1000,
        )
        network.heads.head = nn.Identity()
        return TorchvisionBackbone(network, 384)
    raise ValueError(
        "unsupported backbone; expected resnet18_cifar, resnet18_imagenet, "
        "resnet50_imagenet, convnext_tiny, vgg16_bn, vit_s_16, or vit_b_16"
    )
