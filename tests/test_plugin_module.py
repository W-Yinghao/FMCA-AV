"""Closure plug-in module: both rows build, backbone isolation holds."""

import unittest

import torch

from fmca_av.certificate.block_walk import block_count, blockwise_pooled_features
from fmca_av.certificate.plugin_module import PluginSSLModule
from fmca_av.resnet import resnet18_cifar


def _config(base_name: str, enabled: bool) -> dict:
    return {
        "model": {
            "backbone_width": 16,
            "level_stages": [1, 2, 3],
            "feature_dim": 12,
            "head_hidden_dims": [32],
            "activation": "gelu",
        },
        "base": {"name": base_name, "projector_hidden": 32, "projector_dim": 16},
        "plugin": {"enabled": enabled},
        "loss": {"alpha": 0.2, "beta": 32.0, "gamma": 1.0, "epsilon": 1e-6,
                 "ridge": 1e-3, "whitening_mode": "differentiable",
                 "ema_target_momentum": 0.99},
        "optimizer": {"name": "adamw", "learning_rate": 1e-3},
        "trainer": {"max_epochs": 1},
    }


def _batch(batch_size: int = 6, views: int = 4, endpoint: int = 4) -> dict:
    generator = torch.Generator().manual_seed(0)

    def images(*shape):
        return torch.randn(*shape, 3, 32, 32, generator=generator)

    return {
        "chain": [images(batch_size), images(batch_size), images(batch_size)],
        "children": [images(batch_size, views), images(batch_size, views)],
        "endpoint": images(batch_size, endpoint),
    }


class PluginModuleTests(unittest.TestCase):
    def test_every_base_objective_runs_both_rows(self) -> None:
        batch = _batch()
        for base_name in ("barlow_twins", "vicreg", "frossl"):
            for enabled in (False, True):
                module = PluginSSLModule(_config(base_name, enabled))
                total = module._shared_step(batch, "train")
                self.assertTrue(torch.isfinite(total), msg=f"{base_name}/{enabled}")

    def test_base_row_backbone_sees_only_the_base_loss(self) -> None:
        batch = _batch()
        module = PluginSSLModule(_config("barlow_twins", enabled=False))
        _, metrics = module._hierarchy_terms(batch, detach_backbone=True)
        terms, _ = module._hierarchy_terms(batch, detach_backbone=True)
        terms.backward()
        # Hierarchy heads receive gradients...
        head_gradient = next(module.projectors[0].parameters()).grad
        self.assertIsNotNone(head_gradient)
        self.assertGreater(float(head_gradient.abs().max()), 0.0)
        # ...but the backbone does not (base row isolation).
        stem_gradient = next(module.backbone.stem.parameters()).grad
        self.assertTrue(stem_gradient is None or float(stem_gradient.abs().max()) == 0.0)

    def test_plugin_row_routes_closure_into_the_backbone(self) -> None:
        batch = _batch()
        module = PluginSSLModule(_config("barlow_twins", enabled=True))
        terms, _ = module._hierarchy_terms(batch, detach_backbone=False)
        terms.backward()
        stem_gradient = next(module.backbone.stem.parameters()).grad
        self.assertIsNotNone(stem_gradient)
        self.assertGreater(float(stem_gradient.abs().max()), 0.0)


class BlockWalkTests(unittest.TestCase):
    def test_cifar_resnet_block_walk_shapes(self) -> None:
        model = resnet18_cifar(width=16)
        images = torch.randn(2, 3, 32, 32)
        features = blockwise_pooled_features(model, images)
        self.assertEqual(len(features), block_count(model))
        self.assertEqual(len(features), 1 + 8)  # stem + 2 blocks x 4 stages
        for tensor in features:
            self.assertEqual(tensor.shape[0], 2)
            self.assertEqual(tensor.ndim, 2)


if __name__ == "__main__":
    unittest.main()
