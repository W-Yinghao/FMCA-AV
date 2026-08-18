"""Hierarchy gate module: all seven variants build, run, and backpropagate
on one shared synthetic nested-tree batch."""

import unittest

import torch

from fmca_av.certificate.estimation import ChainFeatureBatch
from fmca_av.certificate.hierarchy_module import GATE_VARIANTS, HierarchyCertificateModule


def _config(variant: str) -> dict:
    return {
        "model": {
            "backbone_width": 16,
            "level_stages": [1, 2, 3],
            "feature_dim": 12,
            "head_hidden_dims": [32],
            "activation": "gelu",
        },
        "variant": variant,
        "loss": {"alpha": 0.0, "beta": 1.0, "gamma": 1.0, "epsilon": 1e-6},
        "optimizer": {"name": "adamw", "learning_rate": 1e-3},
        "trainer": {"max_epochs": 1},
    }


def _batch(batch_size: int = 6, views: int = 4, endpoint: int = 2) -> dict:
    generator = torch.Generator().manual_seed(0)

    def images(*shape):
        return torch.randn(*shape, 3, 32, 32, generator=generator)

    return {
        "chain": [images(batch_size), images(batch_size), images(batch_size)],
        "children": [images(batch_size, views), images(batch_size, views)],
        "endpoint": images(batch_size, endpoint),
    }


class HierarchyModuleTests(unittest.TestCase):
    def test_every_variant_produces_a_finite_loss(self) -> None:
        batch = _batch()
        for variant in GATE_VARIANTS:
            module = HierarchyCertificateModule(_config(variant))
            features = module.feature_batch(batch)
            total, metrics = module._variant_loss(features)
            self.assertTrue(torch.isfinite(total), msg=variant)
            for name, value in metrics.items():
                self.assertTrue(torch.isfinite(torch.tensor(value)), msg=f"{variant}/{name}")

    def test_full_method_backpropagates_into_backbone_and_all_projectors(self) -> None:
        module = HierarchyCertificateModule(_config("product_endpoint"))
        features = module.feature_batch(_batch())
        total, _ = module._variant_loss(features)
        total.backward()
        stem_gradient = next(module.backbone.stem.parameters()).grad
        self.assertIsNotNone(stem_gradient)
        self.assertGreater(float(stem_gradient.abs().max()), 0.0)
        for level, projector in enumerate(module.projectors):
            gradient = next(projector.parameters()).grad
            self.assertIsNotNone(gradient, msg=f"projector {level}")
            self.assertGreater(float(gradient.abs().max()), 0.0, msg=f"projector {level}")

    def test_levels_share_one_projector_for_chain_and_children(self) -> None:
        module = HierarchyCertificateModule(_config("product_endpoint"))
        batch = _batch()
        features = module.feature_batch(batch)
        # Children of edge l live at level l+1 and must be encoded by the
        # SAME projector as the chain state of level l+1.
        manual = module.encode_level(batch["children"][0], 1)
        self.assertTrue(torch.allclose(features.children[0], manual, atol=1e-6))

    def test_feature_shapes_follow_the_tree_layout(self) -> None:
        module = HierarchyCertificateModule(_config("product_endpoint"))
        features = module.feature_batch(_batch())
        self.assertEqual(tuple(features.chain[0].shape), (6, 12))
        self.assertEqual(tuple(features.children[0].shape), (6, 4, 12))
        self.assertEqual(tuple(features.endpoint_descendants.shape), (6, 2, 12))

    def test_variant_losses_discriminate_nilpotent_structure(self) -> None:
        """Additive rows must reward strong local edges even when the ordered
        product is zero; the product row must not.  Feed the nilpotent
        counterexample's features directly through the loss assemblies."""

        from fmca_av.certificate.counterexamples import case_nilpotent

        batch = case_nilpotent().sample(
            4000, children_per_edge=4, endpoint_descendants=4,
            generator=torch.Generator().manual_seed(9),
        )
        features = ChainFeatureBatch(
            chain=[states.float() for states in batch.chain],
            children=[descendants.float() for descendants in batch.children],
            endpoint_descendants=batch.endpoint_descendants.float(),
        )
        additive = HierarchyCertificateModule(_config("additive_mview"))
        product = HierarchyCertificateModule(_config("product_only"))
        _, additive_metrics = additive._variant_loss(features)
        _, product_metrics = product._variant_loss(features)
        # Dimension-normalized per-edge scores (1.0/2 per nilpotent edge),
        # shrunk by the 0.1 train ridge.
        self.assertGreater(additive_metrics["edge_score_sum"], 0.7)
        self.assertLess(product_metrics["product_score"], 0.05)

    def test_decreasing_or_repeated_level_stages_are_rejected(self) -> None:
        for stages in ([3, 2, 1], [2, 2, 3]):
            config = _config("product_endpoint")
            config["model"]["level_stages"] = stages
            with self.assertRaises(ValueError):
                HierarchyCertificateModule(config)

    def test_unknown_variant_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HierarchyCertificateModule(_config("telescoping"))

    def test_faithful_trace_flat_recipe_builds_f_head_and_backpropagates(self) -> None:
        config = _config("final_mview")
        config["loss"]["flat_recipe"] = "faithful_trace"
        module = HierarchyCertificateModule(config)
        self.assertIsNotNone(module.flat_f_head)
        features = module.feature_batch(_batch())
        total, metrics = module._variant_loss(features)
        self.assertTrue(torch.isfinite(total))
        self.assertIn("flat_trace_score", metrics)
        total.backward()
        gradient = next(module.flat_f_head.parameters()).grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().max()), 0.0)
        # Hierarchy variants must NOT allocate the flat head.
        hierarchy = HierarchyCertificateModule(
            {**config, "variant": "product_endpoint"}
        )
        self.assertIsNone(hierarchy.flat_f_head)

    def test_faithful_trace_additive_matches_per_edge_estimator(self) -> None:
        from fmca_av.objectives import trace_score
        from fmca_av.operators import estimate_moments

        config = _config("additive_mview")
        config["loss"]["additive_recipe"] = "faithful_trace"
        module = HierarchyCertificateModule(config)
        features = module.feature_batch(_batch())
        total, metrics = module._variant_loss(features)
        self.assertIn("edge_trace_sum", metrics)
        expected = sum(
            float(trace_score(estimate_moments(features.chain[e], features.children[e], centered=True), ridge=1e-3))
            for e in range(2)
        )
        self.assertAlmostEqual(float(-total), expected, places=4)
        total.backward()
        stem_gradient = next(module.backbone.stem.parameters()).grad
        self.assertIsNotNone(stem_gradient)
        self.assertGreater(float(stem_gradient.abs().max()), 0.0)

    def test_faithful_trace_matches_the_formal_estimator(self) -> None:
        from fmca_av.objectives import trace_score
        from fmca_av.operators import estimate_moments

        config = _config("final_2view")
        config["loss"]["flat_recipe"] = "faithful_trace"
        module = HierarchyCertificateModule(config)
        features = module.feature_batch(_batch())
        total, metrics = module._variant_loss(features)
        leaf = features.endpoint_descendants[:, :2]
        f_features = module.flat_f_head(leaf.mean(dim=1))
        expected = trace_score(estimate_moments(f_features, leaf, centered=True), ridge=1e-3)
        self.assertAlmostEqual(float(-total), float(expected), places=5)


if __name__ == "__main__":
    unittest.main()
