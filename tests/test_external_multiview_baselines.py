import json
from pathlib import Path
import unittest

import torch
import torch.nn.functional as functional

from fmca_av.baselines import (
    BaselineSSL,
    fastssl_barlow_twins_loss,
    fastssl_vicreg_loss,
    frossl_loss,
    frossl_loss_components,
)
from fmca_av.backbones import build_backbone
from fmca_av.data.cifar import HAIHierarchicalDataset, HAIViewTransform


ROOT = Path(__file__).resolve().parents[1]


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


class ExternalMultiviewBaselineTests(unittest.TestCase):
    def test_fastssl_barlow_matches_pinned_multiview_formula(self) -> None:
        torch.manual_seed(12)
        values = torch.randn(7, 4, 5, dtype=torch.float64, requires_grad=True)
        actual = fastssl_barlow_twins_loss(values, 0.2)
        normalized = functional.normalize(values, dim=-1).reshape(-1, 5)
        normalized = (normalized - normalized.mean(0)) / normalized.std(0)
        normalized = normalized.reshape(7, 4, 5)
        mean = normalized.mean(1)
        diagonal = ((normalized * mean[:, None]).mean(0) - 1).square().sum() / 4
        correlation = mean.T @ mean / 7
        expected = diagonal + 0.2 * off_diagonal(correlation).square().sum()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
        actual.backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_fastssl_vicreg_uses_every_view_and_other_view_mean(self) -> None:
        torch.manual_seed(21)
        values = torch.randn(9, 4, 6, dtype=torch.float64)
        base = fastssl_vicreg_loss(values, 25.0, 25.0)
        changed = values.clone()
        changed[:, 3] += torch.linspace(-2, 2, 9, dtype=values.dtype)[:, None]
        altered = fastssl_vicreg_loss(changed, 25.0, 25.0)
        self.assertFalse(torch.allclose(base, altered))
        self.assertTrue(torch.isfinite(base))

    def test_frossl_matches_official_multiview_linear_objective(self) -> None:
        torch.manual_seed(33)
        values = torch.randn(8, 3, 5, dtype=torch.float64, requires_grad=True)
        actual = frossl_loss(values, 1.0)
        normalized = functional.normalize(values, p=2, dim=0)
        average = normalized.mean(1)
        expected = values.new_zeros(())
        for index in range(3):
            view = normalized[:, index]
            gram = view.T @ view
            gram = gram / gram.trace().detach()
            expected = expected + 2 * torch.log(torch.linalg.matrix_norm(gram, ord="fro"))
            expected = expected + 3 * functional.mse_loss(view, average) * 5
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
        actual.backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_frossl_components_sum_to_exact_objective(self) -> None:
        torch.manual_seed(41)
        values = torch.randn(6, 8, 7, dtype=torch.float64)
        total, invariance, regularization = frossl_loss_components(values, 2.0)
        self.assertTrue(torch.allclose(total, invariance + regularization))
        self.assertTrue(torch.allclose(total, frossl_loss(values, 2.0)))
        self.assertGreater(float(invariance), 0.0)
        self.assertLessEqual(float(regularization), 0.0)

    def test_frossl_sequential_forward_preserves_view_axis(self) -> None:
        config = {
            "experiment": {"method": "frossl"},
            "model": {
                "backbone": "resnet18_cifar", "backbone_width": 4,
                "projection_hidden_dim": 8, "projection_dim": 6,
                "view_forward_mode": "sequential",
            },
            "objective": {"invariance_weight": 2.0},
            "optimizer": {"name": "sgd", "learning_rate": 0.05, "scheduler": "none"},
            "trainer": {"max_epochs": 1},
        }
        model = BaselineSSL(config)
        model.log = lambda *args, **kwargs: None
        views = torch.randn(3, 4, 3, 32, 32)
        loss = model._shared_step((views, torch.zeros(3, dtype=torch.long), torch.arange(3)), "train")
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.view_forward_mode, "sequential")

    def test_frossl_official_online_classifier_is_backbone_detached(self) -> None:
        config = {
            "experiment": {"method": "frossl"},
            "data": {"dataset": "cifar10"},
            "model": {
                "backbone": "resnet18_cifar", "backbone_width": 4,
                "projection_hidden_dim": 8, "projection_dim": 6,
                "view_forward_mode": "sequential",
            },
            "objective": {"invariance_weight": 1.0, "online_classifier": True},
            "optimizer": {"name": "sgd", "learning_rate": 0.05, "scheduler": "none"},
            "trainer": {"max_epochs": 1},
        }
        model = BaselineSSL(config)
        self.assertIsNotNone(model.online_classifier)
        features = torch.randn(5, model.backbone.output_dim, requires_grad=True)
        labels = torch.arange(5) % 10
        classifier_loss = torch.nn.functional.cross_entropy(
            model.online_classifier(features.detach()), labels
        )
        classifier_loss.backward()
        self.assertIsNone(features.grad)
        self.assertIsNotNone(model.online_classifier.weight.grad)
        optimizer = model.configure_optimizers()
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 0.1)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)

    def test_frossl_official_cifar_stem_matches_pinned_repository(self) -> None:
        backbone = build_backbone("resnet18_frossl_cifar", width=64)
        self.assertEqual(backbone.network.conv1.kernel_size, (3, 3))
        self.assertEqual(backbone.network.conv1.stride, (1, 1))
        self.assertEqual(backbone.network.conv1.padding, (2, 2))
        self.assertIsInstance(backbone.network.maxpool, torch.nn.Identity)

    def test_formal_configs_pin_defining_mechanisms(self) -> None:
        expected = {
            "cifar10_fastssl_barlow_twins.json": ("fastssl_barlow_twins", 256, "adam"),
            "cifar10_fastssl_vicreg.json": ("fastssl_vicreg", 256, "adam"),
            "cifar10_frossl.json": ("frossl", 1024, "lars"),
            "cifar10_hai_simsiam.json": ("hai_simsiam", 2048, "sgd"),
        }
        for filename, (method, projection, optimizer) in expected.items():
            payload = json.loads((ROOT / "configs" / "ssl" / filename).read_text(encoding="utf-8"))
            self.assertEqual(payload["data"]["dataset"], "cifar10")
            self.assertEqual(payload["data"]["num_views"], 8)
            self.assertEqual(payload["experiment"]["method"], method)
            self.assertEqual(payload["model"]["projection_dim"], projection)
            self.assertEqual(payload["optimizer"]["name"], optimizer)

    def test_hai_add_one_dataset_returns_four_adjacent_pairs_and_parameters(self) -> None:
        class OneImage(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int):
                image = (torch.arange(3 * 32 * 32).reshape(3, 32, 32) % 256).to(torch.uint8)
                return image.numpy(), 4

        transform = HAIViewTransform({
            "size": 32,
            "min_scale": 0.5,
            "color_jitter_probability": 1.0,
            "color_jitter_strength": 0.5,
            "grayscale_probability": 1.0,
            "gaussian_blur_probability": 1.0,
            "horizontal_flip_probability": 1.0,
        })
        dataset = HAIHierarchicalDataset(OneImage(), transform, deterministic_seed=17)
        views, label, index, parameters = dataset[0]
        self.assertEqual(views.shape, (8, 3, 32, 32))
        self.assertEqual(parameters.shape, (8, 4))
        self.assertTrue(torch.isfinite(views).all())
        self.assertTrue(torch.isfinite(parameters).all())
        self.assertEqual((label, index), (4, 0))

    def test_hai_uses_all_stage_pairs_and_sums_stage_losses(self) -> None:
        config = {
            "experiment": {"method": "hai_simsiam"},
            "model": {
                "backbone": "resnet18_cifar", "backbone_width": 4,
                "augmentation_embedding_dim": 4, "projection_hidden_dim": 8,
                "projection_dim": 6, "predictor_hidden_dim": 5,
            },
            "objective": {},
            "optimizer": {"name": "sgd", "learning_rate": 0.05, "scheduler": "none"},
            "trainer": {"max_epochs": 1},
        }
        model = BaselineSSL(config)
        model.log = lambda *args, **kwargs: None
        views = torch.randn(3, 8, 3, 32, 32, requires_grad=True)
        parameters = torch.randn(3, 8, 4)
        loss, stage_losses = model.hai_heads.loss(model.backbone, views, parameters)
        self.assertEqual(len(stage_losses), 4)
        self.assertTrue(torch.allclose(loss, torch.stack(stage_losses).sum()))
        loss.backward()
        for stage in range(4):
            pair_gradient = views.grad[:, 2 * stage:2 * stage + 2]
            self.assertGreater(float(pair_gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
