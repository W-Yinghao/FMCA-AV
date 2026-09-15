"""Nested stochastic view tree for the depth-aligned hierarchy.

The generative chain is X0 -> X1 -> ... -> XL where every child is a random
refinement of the REALIZED parent view: a local crop is sampled inside the
realized global crop, a mask is applied to the realized local crop.  This
makes the chain Markov by construction: p(local | global, X) = p(local | global).

Forbidden (and provided only as the ``parallel`` negative-control mode):
sampling M independent augmentations of the original image at every level.
That star layout draws every view from p(Y | X0) and destroys the per-edge
operator semantics.

Chain-state convention: chain state l+1 IS descendant 0 of chain state l,
so the realized path and the conditional descendants stay consistent.

Every crop records its box both in the parent frame and composed back to
the original pixel frame; the containment unit test addresses boxes by
these real coordinates, never by a bench index.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class CropLevelSpec:
    kind: str = "crop"
    min_scale: float = 0.3
    max_scale: float = 0.8
    size: int = 32
    noise_std: float = 0.0
    color_jitter_probability: float = 0.0
    color_jitter_strength: float = 0.5
    color_jitter_hue: bool = False
    grayscale_probability: float = 0.0
    flip_probability: float = 0.0


@dataclass
class MaskLevelSpec:
    kind: str = "mask"
    patch_fraction: float = 0.35
    patches: int = 1
    noise_std: float = 0.0
    grayscale_probability: float = 0.0
    blur_probability: float = 0.0


LevelSpec = object


@dataclass
class ViewTreeConfig:
    root_spec: CropLevelSpec = field(default_factory=lambda: CropLevelSpec(min_scale=0.5, max_scale=1.0))
    edge_specs: List[LevelSpec] = field(
        default_factory=lambda: [CropLevelSpec(min_scale=0.3, max_scale=0.8), MaskLevelSpec()]
    )
    children_per_edge: int = 4
    endpoint_descendants: int = 1
    flip_probability: float = 0.5
    mode: str = "nested"
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

    def __post_init__(self) -> None:
        if self.mode not in {"nested", "parallel"}:
            raise ValueError("mode must be nested or parallel (parallel is the negative control)")
        if len(self.edge_specs) < 1:
            raise ValueError("at least one edge is required")
        if self.children_per_edge < 1 or self.endpoint_descendants < 1:
            raise ValueError("children_per_edge and endpoint_descendants must be positive")


@dataclass
class RealizedView:
    """A realized view: normalized pixels plus its box in both frames.

    ``flipped`` records whether this view's pixel frame is horizontally
    mirrored relative to the original image (a root-level flip propagates
    to every descendant); box composition must undo the mirror so that
    ``box_in_original`` always addresses REAL original-image coordinates.
    """

    pixels: Tensor
    box_in_parent: Tensor
    box_in_original: Tensor
    flipped: bool = False


def _uniform(generator: Optional[torch.Generator], low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


def _sample_crop_box(
    height: int,
    width: int,
    spec: CropLevelSpec,
    generator: Optional[torch.Generator],
) -> Tuple[int, int, int, int]:
    area = height * width
    for _ in range(10):
        target = area * _uniform(generator, spec.min_scale, spec.max_scale)
        log_ratio = _uniform(generator, math.log(3.0 / 4.0), math.log(4.0 / 3.0))
        aspect = math.exp(log_ratio)
        crop_width = int(round(math.sqrt(target * aspect)))
        crop_height = int(round(math.sqrt(target / aspect)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = int(torch.randint(0, height - crop_height + 1, (), generator=generator))
            left = int(torch.randint(0, width - crop_width + 1, (), generator=generator))
            return top, left, crop_height, crop_width
    return 0, 0, height, width


def _resize(pixels: Tensor, size: int) -> Tensor:
    return torch.nn.functional.interpolate(
        pixels.unsqueeze(0), size=(size, size), mode="bicubic", align_corners=False
    ).squeeze(0).clamp(0.0, 1.0)


_LUMA = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)


def _luminance(pixels: Tensor) -> Tensor:
    return (pixels * _LUMA).sum(dim=0, keepdim=True)


def _color_jitter(
    pixels: Tensor,
    strength: float,
    generator: Optional[torch.Generator],
    with_hue: bool = False,
) -> Tensor:
    """Tensor-native brightness/contrast/saturation(/hue) jitter.

    Ranges match the historical CIFARViewTransform: b/c/s in
    [1 - 0.8s, 1 + 0.8s], hue shift in [-0.2s, 0.2s] of the hue circle
    (applied via torchvision's tensor adjust_hue).
    """

    brightness = _uniform(generator, 1 - 0.8 * strength, 1 + 0.8 * strength)
    contrast = _uniform(generator, 1 - 0.8 * strength, 1 + 0.8 * strength)
    saturation = _uniform(generator, 1 - 0.8 * strength, 1 + 0.8 * strength)
    jittered = pixels * brightness
    gray_mean = _luminance(jittered).mean()
    jittered = gray_mean + (jittered - gray_mean) * contrast
    gray = _luminance(jittered)
    jittered = gray + (jittered - gray) * saturation
    jittered = jittered.clamp(0.0, 1.0)
    if with_hue:
        from torchvision.transforms.functional import adjust_hue

        hue_shift = _uniform(generator, -0.2 * strength, 0.2 * strength)
        jittered = adjust_hue(jittered, hue_shift).clamp(0.0, 1.0)
    return jittered


def _grayscale(pixels: Tensor) -> Tensor:
    return _luminance(pixels).expand_as(pixels).clone()


def _gaussian_blur(pixels: Tensor, sigma: float = 1.0) -> Tensor:
    radius = 2
    positions = torch.arange(-radius, radius + 1, dtype=pixels.dtype)
    kernel_1d = torch.exp(-positions.square() / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(3, 1, -1, -1)
    padded = torch.nn.functional.pad(pixels.unsqueeze(0), [radius] * 4, mode="reflect")
    return torch.nn.functional.conv2d(padded, kernel, groups=3).squeeze(0)


def _compose_box(
    parent_box_in_original: Tensor,
    parent_size: Tuple[int, int],
    child_box: Tensor,
    parent_flipped: bool,
) -> Tensor:
    """Map a child crop box from the parent's pixel frame to the original frame.

    A flipped parent frame is horizontally mirrored relative to the original
    image, so the child's horizontal offset must be mirrored back before
    scaling into original coordinates.
    """

    parent_top, parent_left, parent_height, parent_width = parent_box_in_original.tolist()
    frame_height, frame_width = parent_size
    top, left, height, width = child_box.tolist()
    if parent_flipped:
        left = frame_width - left - width
    scale_y = parent_height / frame_height
    scale_x = parent_width / frame_width
    return torch.tensor(
        [
            parent_top + top * scale_y,
            parent_left + left * scale_x,
            height * scale_y,
            width * scale_x,
        ],
        dtype=torch.float32,
    )


class NestedViewTreeSampler:
    """Samples one nested view tree from a raw [3, H, W] uint8 image."""

    def __init__(self, config: ViewTreeConfig) -> None:
        self.config = config
        self._mean = torch.tensor(config.mean).view(3, 1, 1)
        self._std = torch.tensor(config.std).view(3, 1, 1)

    @property
    def num_levels(self) -> int:
        return len(self.config.edge_specs) + 1

    def _normalize(self, pixels: Tensor) -> Tensor:
        return (pixels - self._mean) / self._std

    def _root_view(self, image: Tensor, generator: Optional[torch.Generator]) -> RealizedView:
        spec = self.config.root_spec
        height, width = image.shape[-2:]
        top, left, crop_height, crop_width = _sample_crop_box(height, width, spec, generator)
        pixels = _resize(image[:, top:top + crop_height, left:left + crop_width], spec.size)
        flipped = float(torch.rand((), generator=generator)) < self.config.flip_probability
        if flipped:
            pixels = pixels.flip(-1)
        pixels = self._apply_noise(pixels, spec.noise_std, generator)
        box = torch.tensor([top, left, crop_height, crop_width], dtype=torch.float32)
        return RealizedView(pixels=pixels, box_in_parent=box, box_in_original=box, flipped=flipped)

    def _apply_noise(
        self, pixels: Tensor, noise_std: float, generator: Optional[torch.Generator]
    ) -> Tensor:
        if noise_std <= 0:
            return pixels
        noise = torch.randn(pixels.shape, generator=generator, dtype=pixels.dtype)
        return (pixels + noise_std * noise).clamp(0.0, 1.0)

    def _child_view(
        self,
        parent: RealizedView,
        spec: LevelSpec,
        generator: Optional[torch.Generator],
    ) -> RealizedView:
        source = parent.pixels
        height, width = source.shape[-2:]
        if isinstance(spec, CropLevelSpec):
            top, left, crop_height, crop_width = _sample_crop_box(height, width, spec, generator)
            pixels = _resize(source[:, top:top + crop_height, left:left + crop_width], spec.size)
            # Per-child photometric/geometric refinement, conditional on the
            # realized parent pixels: legitimate stochastic channels.
            child_flip = float(torch.rand((), generator=generator)) < spec.flip_probability
            if child_flip:
                pixels = pixels.flip(-1)
            if float(torch.rand((), generator=generator)) < spec.color_jitter_probability:
                pixels = _color_jitter(
                    pixels, spec.color_jitter_strength, generator, with_hue=spec.color_jitter_hue
                )
            if float(torch.rand((), generator=generator)) < spec.grayscale_probability:
                pixels = _grayscale(pixels)
            pixels = self._apply_noise(pixels, spec.noise_std, generator)
            box = torch.tensor([top, left, crop_height, crop_width], dtype=torch.float32)
            # The child's own frame parity flips relative to the original
            # whenever it flips relative to its parent.
            return RealizedView(
                pixels=pixels,
                box_in_parent=box,
                box_in_original=_compose_box(
                    parent.box_in_original, (height, width), box, parent.flipped
                ),
                flipped=parent.flipped != child_flip,
            )
        if isinstance(spec, MaskLevelSpec):
            # A mask child occupies the FULL parent frame.
            full_frame = torch.tensor([0.0, 0.0, float(height), float(width)], dtype=torch.float32)
            pixels = source.clone()
            fill = source.mean(dim=(-2, -1), keepdim=True)
            side = max(1, int(round(math.sqrt(spec.patch_fraction / spec.patches) * height)))
            for _ in range(spec.patches):
                top = int(torch.randint(0, height - side + 1, (), generator=generator))
                left = int(torch.randint(0, width - side + 1, (), generator=generator))
                pixels[:, top:top + side, left:left + side] = fill
            if float(torch.rand((), generator=generator)) < spec.grayscale_probability:
                pixels = _grayscale(pixels)
            if float(torch.rand((), generator=generator)) < spec.blur_probability:
                pixels = _gaussian_blur(pixels)
            pixels = self._apply_noise(pixels, spec.noise_std, generator)
            return RealizedView(
                pixels=pixels,
                box_in_parent=full_frame,
                box_in_original=parent.box_in_original.clone(),
                flipped=parent.flipped,
            )
        raise TypeError(f"unsupported level spec {type(spec)!r}")

    def sample(
        self, image: Tensor, generator: Optional[torch.Generator] = None
    ) -> Dict[str, object]:
        if image.ndim != 3:
            raise ValueError("image must be [channels, height, width]")
        nested = self.config.mode == "nested"
        root = self._root_view(image, generator)
        chain: List[RealizedView] = [root]
        children_views: List[List[RealizedView]] = []
        for spec in self.config.edge_specs:
            parent = chain[-1]
            siblings = []
            for _ in range(self.config.children_per_edge):
                # Parallel mode ignores the realized parent and redraws from
                # the original image: the forbidden star construction.
                source = self._root_view(image, generator) if not nested else parent
                siblings.append(self._child_view(source, spec, generator))
            children_views.append(siblings)
            chain.append(siblings[0])
        endpoint_views: List[RealizedView] = []
        for _ in range(self.config.endpoint_descendants):
            state = root
            for spec in self.config.edge_specs:
                state = self._child_view(state, spec, generator)
            endpoint_views.append(state)
        return {
            "chain": [self._normalize(view.pixels) for view in chain],
            "children": [
                torch.stack([self._normalize(view.pixels) for view in siblings])
                for siblings in children_views
            ],
            "endpoint": torch.stack([self._normalize(view.pixels) for view in endpoint_views]),
            "boxes_chain": torch.stack([view.box_in_original for view in chain]),
            "boxes_children": [
                torch.stack([view.box_in_original for view in siblings])
                for siblings in children_views
            ],
            "boxes_endpoint": torch.stack([view.box_in_original for view in endpoint_views]),
        }


class NestedViewTreeDataset(Dataset):
    """Wraps a base dataset of raw [3, H, W] uint8 images into view trees."""

    def __init__(
        self,
        base: Dataset,
        config: ViewTreeConfig,
        deterministic_seed: Optional[int] = None,
    ) -> None:
        self.base = base
        self.sampler = NestedViewTreeSampler(config)
        self.deterministic_seed = deterministic_seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image, label = self.base[index]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.astype(np.float32)) / 255.0
        generator = None
        if self.deterministic_seed is not None:
            generator = torch.Generator().manual_seed(self.deterministic_seed + index)
        tree = self.sampler.sample(image, generator)
        tree["label"] = int(label)
        tree["index"] = int(index)
        return tree
