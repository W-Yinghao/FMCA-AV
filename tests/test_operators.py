import unittest

import torch

from fmca_av.objectives import trace_score
from fmca_av.operators import estimate_moments, fit_spectral_calibration


class OperatorEstimatorTests(unittest.TestCase):
    def test_rg_is_mean_of_outer_products_not_outer_product_of_mean(self) -> None:
        f = torch.tensor([[1.0], [-1.0]])
        g_views = torch.tensor([[[1.0], [-1.0]], [[2.0], [-2.0]]])
        moments = estimate_moments(f, g_views, centered=False)
        wrong_rg = torch.einsum("bk,bl->kl", g_views.mean(dim=1), g_views.mean(dim=1)) / 2
        self.assertAlmostEqual(float(moments.r_g), 2.5)
        self.assertAlmostEqual(float(wrong_rg), 0.0)

    def test_cross_moment_uses_conditional_view_mean(self) -> None:
        f = torch.tensor([[1.0], [3.0]])
        g_views = torch.tensor([[[2.0], [4.0]], [[1.0], [5.0]]])
        moments = estimate_moments(f, g_views, centered=False)
        self.assertAlmostEqual(float(moments.p_fg), 6.0)

    def test_trace_matches_squared_calibrated_singular_values(self) -> None:
        generator = torch.Generator().manual_seed(11)
        f = torch.randn(2048, 3, generator=generator, dtype=torch.float64)
        noise = torch.randn(2048, 4, 3, generator=generator, dtype=torch.float64)
        g = 0.7 * f.unsqueeze(1) + (1.0 - 0.7 ** 2) ** 0.5 * noise
        moments = estimate_moments(f, g, centered=True)
        calibration = fit_spectral_calibration(f, g, ridge=1e-3, centered=True)
        self.assertTrue(torch.allclose(trace_score(moments), calibration.eigenvalues.sum()))

    def test_relative_ridge_is_invariant_to_global_feature_scale(self) -> None:
        generator = torch.Generator().manual_seed(17)
        f = 3.0 * torch.randn(1024, 2, generator=generator, dtype=torch.float64)
        g = 0.6 * f.unsqueeze(1) + torch.randn(1024, 3, 2, generator=generator, dtype=torch.float64)
        base = trace_score(estimate_moments(f, g), ridge=1e-3)
        scaled = trace_score(estimate_moments(100.0 * f, 100.0 * g), ridge=1e-3)
        self.assertTrue(torch.allclose(base, scaled, atol=1e-10, rtol=1e-10))


if __name__ == "__main__":
    unittest.main()
