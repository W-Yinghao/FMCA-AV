"""Nested view tree: Markov-by-construction nesting, real-coordinate box
containment, chain-state convention, determinism, and the parallel (star)
negative-control mode."""

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fmca_av.data.view_tree import (
    CropLevelSpec,
    MaskLevelSpec,
    NestedViewTreeDataset,
    NestedViewTreeSampler,
    ViewTreeConfig,
)


class SyntheticImages(Dataset):
    def __init__(self, count: int = 16, size: int = 64, seed: int = 0) -> None:
        generator = np.random.default_rng(seed)
        self.images = generator.integers(0, 256, size=(count, 3, size, size), dtype=np.uint8)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        return self.images[index], index % 10


def _default_config(mode: str = "nested") -> ViewTreeConfig:
    return ViewTreeConfig(
        root_spec=CropLevelSpec(min_scale=0.5, max_scale=1.0, size=32),
        edge_specs=[CropLevelSpec(min_scale=0.3, max_scale=0.8, size=32), MaskLevelSpec()],
        children_per_edge=3,
        endpoint_descendants=2,
        mode=mode,
    )


def _box_inside(child: torch.Tensor, parent: torch.Tensor, tolerance: float = 1e-3) -> bool:
    child_top, child_left, child_height, child_width = child.tolist()
    parent_top, parent_left, parent_height, parent_width = parent.tolist()
    return (
        child_top >= parent_top - tolerance
        and child_left >= parent_left - tolerance
        and child_top + child_height <= parent_top + parent_height + tolerance
        and child_left + child_width <= parent_left + parent_width + tolerance
    )


class NestedTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = NestedViewTreeDataset(SyntheticImages(), _default_config(), deterministic_seed=100)

    def test_every_nested_child_box_lies_inside_its_parent_box(self) -> None:
        for index in range(len(self.dataset)):
            tree = self.dataset[index]
            for edge, boxes in enumerate(tree["boxes_children"]):
                parent_box = tree["boxes_chain"][edge]
                for view in range(boxes.shape[0]):
                    self.assertTrue(
                        _box_inside(boxes[view], parent_box),
                        msg=f"index {index} edge {edge} view {view}: {boxes[view]} vs {parent_box}",
                    )

    def test_endpoint_descendants_stay_inside_the_realized_root(self) -> None:
        """Endpoint views must descend from the REALIZED root view, never
        from a fresh root crop of the original image."""

        for index in range(len(self.dataset)):
            tree = self.dataset[index]
            root_box = tree["boxes_chain"][0]
            for view in range(tree["boxes_endpoint"].shape[0]):
                self.assertTrue(
                    _box_inside(tree["boxes_endpoint"][view], root_box),
                    msg=f"index {index} endpoint view {view}",
                )

    def test_chain_state_is_descendant_zero(self) -> None:
        tree = self.dataset[0]
        for edge in range(len(tree["children"])):
            self.assertTrue(torch.equal(tree["chain"][edge + 1], tree["children"][edge][0]))

    def test_mask_level_keeps_the_box_but_changes_pixels(self) -> None:
        tree = self.dataset[1]
        self.assertTrue(torch.allclose(tree["boxes_chain"][2], tree["boxes_chain"][1]))
        difference = (tree["chain"][2] - tree["chain"][1]).abs().sum()
        self.assertGreater(float(difference), 0.0)

    def test_deterministic_seed_reproduces_the_tree(self) -> None:
        other = NestedViewTreeDataset(SyntheticImages(), _default_config(), deterministic_seed=100)
        first, second = self.dataset[3], other[3]
        for level in range(len(first["chain"])):
            self.assertTrue(torch.equal(first["chain"][level], second["chain"][level]))
        different = NestedViewTreeDataset(SyntheticImages(), _default_config(), deterministic_seed=999)[3]
        self.assertFalse(torch.equal(first["chain"][0], different["chain"][0]))

    def test_children_are_distinct_conditional_draws(self) -> None:
        tree = self.dataset[2]
        crops = tree["children"][0]
        self.assertFalse(torch.equal(crops[0], crops[1]))

    def test_batches_collate_with_expected_shapes(self) -> None:
        loader = DataLoader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(len(batch["chain"]), 3)
        self.assertEqual(tuple(batch["chain"][0].shape), (4, 3, 32, 32))
        self.assertEqual(tuple(batch["children"][0].shape), (4, 3, 3, 32, 32))
        self.assertEqual(tuple(batch["endpoint"].shape), (4, 2, 3, 32, 32))


class FlipBookkeepingTests(unittest.TestCase):
    def test_boxes_address_real_original_coordinates_under_flips(self) -> None:
        """A horizontal-gradient probe: the brightness under box_in_original
        in the ORIGINAL image must match the child view's own brightness.
        Mirrored (wrong) box coordinates would land on the opposite side of
        the gradient and systematically break this identity."""

        size = 64
        gradient = np.tile(
            np.linspace(0, 255, size, dtype=np.float64), (size, 1)
        ).astype(np.uint8)
        image = np.stack([gradient] * 3)
        config = ViewTreeConfig(
            root_spec=CropLevelSpec(min_scale=0.5, max_scale=0.9, size=32),
            edge_specs=[CropLevelSpec(min_scale=0.05, max_scale=0.15, size=32)],
            children_per_edge=4,
            endpoint_descendants=1,
            flip_probability=1.0,
        )
        sampler = NestedViewTreeSampler(config)
        mean = torch.tensor(config.mean).view(3, 1, 1)
        std = torch.tensor(config.std).view(3, 1, 1)
        original = torch.from_numpy(image.astype(np.float32)) / 255.0
        worst = 0.0
        for seed in range(10):
            tree = sampler.sample(original, torch.Generator().manual_seed(seed))
            for view in range(4):
                box = tree["boxes_children"][0][view]
                top, left, height, width = box.tolist()
                region = original[
                    :,
                    int(round(top)):max(int(round(top + height)), int(round(top)) + 1),
                    int(round(left)):max(int(round(left + width)), int(round(left)) + 1),
                ]
                child = tree["children"][0][view] * std + mean
                worst = max(worst, abs(float(region.mean()) - float(child.mean())))
        self.assertLess(worst, 0.08)


class ParallelModeTests(unittest.TestCase):
    def test_parallel_mode_escapes_the_realized_parent(self) -> None:
        dataset = NestedViewTreeDataset(
            SyntheticImages(count=24), _default_config(mode="parallel"), deterministic_seed=7
        )
        violations = 0
        for index in range(len(dataset)):
            tree = dataset[index]
            parent_box = tree["boxes_chain"][0]
            boxes = tree["boxes_children"][0]
            for view in range(boxes.shape[0]):
                if not _box_inside(boxes[view], parent_box):
                    violations += 1
        self.assertGreater(violations, 0)


class SamplerValidationTests(unittest.TestCase):
    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ViewTreeConfig(mode="star")

    def test_sampler_requires_channel_first_images(self) -> None:
        sampler = NestedViewTreeSampler(_default_config())
        with self.assertRaises(ValueError):
            sampler.sample(torch.rand(64, 64))


if __name__ == "__main__":
    unittest.main()
