import glob
import json
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from fmca_av.models import MLP

REPO = Path(__file__).resolve().parent.parent


def batchnorms(module) -> int:
    return sum(1 for m in module.modules() if isinstance(m, nn.BatchNorm1d))


class HeadNormalizationTests(unittest.TestCase):
    def test_default_head_is_unchanged(self) -> None:
        """Every arm on disk was trained with a head carrying no norm."""
        head = MLP(128, 128, [512, 512], "gelu")
        self.assertEqual(batchnorms(head), 0)
        self.assertEqual(
            [type(m).__name__ for m in head.network],
            ["Linear", "GELU", "Linear", "GELU", "Linear"])

    def test_batch_normalization_goes_after_each_hidden_linear(self) -> None:
        head = MLP(128, 128, [512, 512], "gelu", "batch")
        self.assertEqual(
            [type(m).__name__ for m in head.network],
            ["Linear", "BatchNorm1d", "GELU",
             "Linear", "BatchNorm1d", "GELU",
             "Linear"])
        # Never after the output layer.
        self.assertIsInstance(head.network[-1], nn.Linear)

    def test_normalized_head_still_maps_to_the_same_shape(self) -> None:
        head = MLP(128, 64, [512], "gelu", "batch").train()
        self.assertEqual(head(torch.randn(8, 128)).shape, (8, 64))

    def test_bad_normalization_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            MLP(8, 8, [8], "gelu", "layer")

    def test_no_config_already_run_gains_a_normalization(self) -> None:
        """The fix must not silently re-specify any existing arm's head."""
        changed = []
        for path in sorted(glob.glob(str(REPO / "configs/**/*.json"), recursive=True)):
            name = Path(path).name
            if name == "cifar10_byol_bn.json":
                continue
            try:
                config = json.loads(Path(path).read_text())
            except Exception:
                continue
            model = config.get("model")
            if isinstance(model, dict) and model.get("head_normalization", "none") != "none":
                changed.append(name)
        self.assertEqual(changed, [])

    def test_the_fixed_byol_config_asks_for_batch_normalization(self) -> None:
        config = json.loads((REPO / "configs/ssl_matched/cifar10_byol_bn.json").read_text())
        self.assertEqual(config["model"]["head_normalization"], "batch")
        self.assertEqual(config["experiment"]["method"], "byol")
        original = json.loads((REPO / "configs/ssl_matched/cifar10_byol.json").read_text())
        # Single-factor: nothing but the head normalization and the labels.
        for section in ("data", "optimizer", "trainer", "objective"):
            self.assertEqual(config[section], original[section], msg=section)
        self.assertEqual(config["seed"], original["seed"])
        differing = {k for k in set(config["model"]) | set(original["model"])
                     if config["model"].get(k) != original["model"].get(k)}
        self.assertEqual(differing, {"head_normalization"})


if __name__ == "__main__":
    unittest.main()
