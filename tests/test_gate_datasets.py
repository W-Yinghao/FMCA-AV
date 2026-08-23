"""The gate's dataset dispatch must not disturb the CIFAR path."""

import unittest

import numpy as np
import torch

from fmca_av.certificate.gate_data import ArrayImageFiles, GateProbeTransform
from fmca_av.certificate.stage_backbone import StageTappedCIFARResNet


class _FakePIL:
    """Minimal stand-in with the PIL surface ArrayImageFiles touches."""

    def __init__(self, size, value):
        self.size = size
        self._value = value

    def resize(self, size, _resample):
        return _FakePIL(size, self._value)

    def __array__(self, dtype=None):
        array = np.full((self.size[1], self.size[0], 3), self._value, dtype=np.uint8)
        return array if dtype is None else array.astype(dtype)


class _FakeFiles:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return _FakePIL(self.size, index + 1), index


class ArrayAdapterTest(unittest.TestCase):
    def test_emits_cifar_layout(self):
        files = ArrayImageFiles(_FakeFiles((64, 64)), 64)
        image, label = files[1]
        self.assertEqual(image.shape, (3, 64, 64))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(label, 1)

    def test_resizes_when_source_differs(self):
        files = ArrayImageFiles(_FakeFiles((80, 80)), 64)
        self.assertEqual(files[0][0].shape, (3, 64, 64))


class ProbeTransformTest(unittest.TestCase):
    def test_preserves_native_size_at_64(self):
        transform = GateProbeTransform(True, [0.5] * 3, [0.5] * 3, 64)
        out = transform(np.zeros((3, 64, 64), dtype=np.uint8))
        self.assertEqual(out.shape, (3, 64, 64))

    def test_eval_mode_is_deterministic(self):
        transform = GateProbeTransform(False, [0.5] * 3, [0.5] * 3, 64)
        array = np.full((3, 64, 64), 128, dtype=np.uint8)
        torch.testing.assert_close(transform(array), transform(array))


class StemTest(unittest.TestCase):
    def test_cifar_stem_unchanged_at_32(self):
        model = StageTappedCIFARResNet(width=16)
        stages = model.forward_stages(torch.zeros(2, 3, 32, 32))
        self.assertEqual(len(stages), 4)
        self.assertEqual(stages[-1].shape, (2, 128))

    def test_downsample_stem_matches_cifar_dims_at_64(self):
        cifar = StageTappedCIFARResNet(width=16)
        wide = StageTappedCIFARResNet(width=16, stem="downsample")
        small = [s.shape for s in cifar.forward_stages(torch.zeros(2, 3, 32, 32))]
        large = [s.shape for s in wide.forward_stages(torch.zeros(2, 3, 64, 64))]
        self.assertEqual(small, large)

    def test_rejects_unknown_stem(self):
        with self.assertRaises(ValueError):
            StageTappedCIFARResNet(stem="resnet50")


if __name__ == "__main__":
    unittest.main()
