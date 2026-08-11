import json
from pathlib import Path
import unittest

import torch
import torch.nn.functional as functional

from fmca_av.baselines import (
    fastssl_barlow_twins_loss,
    fastssl_vicreg_loss,
    frossl_loss,
)


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

    def test_formal_configs_pin_defining_mechanisms(self) -> None:
        expected = {
            "cifar10_fastssl_barlow_twins.json": ("fastssl_barlow_twins", 256, "adam"),
            "cifar10_fastssl_vicreg.json": ("fastssl_vicreg", 256, "adam"),
            "cifar10_frossl.json": ("frossl", 1024, "lars"),
        }
        for filename, (method, projection, optimizer) in expected.items():
            payload = json.loads((ROOT / "configs" / "ssl" / filename).read_text(encoding="utf-8"))
            self.assertEqual(payload["data"]["dataset"], "cifar10")
            self.assertEqual(payload["data"]["num_views"], 8)
            self.assertEqual(payload["experiment"]["method"], method)
            self.assertEqual(payload["model"]["projection_dim"], projection)
            self.assertEqual(payload["optimizer"]["name"], optimizer)


if __name__ == "__main__":
    unittest.main()
