import unittest

import torch

from fmca_av.objectives import logdet_score, trace_score
from fmca_av.operators import (
    clipped_logdet_from_eigenvalues,
    estimate_moments,
    evaluate_heldout_spectrum,
    fit_spectral_calibration,
    relative_ridge_scale,
    whitened_cross_operator,
)


class OperatorEstimatorTests(unittest.TestCase):
    def test_low_precision_moments_accumulate_in_float32_with_gradients(self) -> None:
        f = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]], dtype=torch.float16, requires_grad=True)
        g = f.detach().reshape(2, 1, 2).repeat(1, 2, 1).requires_grad_()
        moments = estimate_moments(f, g, centered=True)
        self.assertEqual(moments.r_f.dtype, torch.float32)
        self.assertEqual(moments.r_g.dtype, torch.float32)
        self.assertEqual(moments.p_fg.dtype, torch.float32)
        self.assertTrue(torch.isfinite(moments.r_f).all())
        self.assertTrue(torch.isfinite(moments.r_g).all())
        score = trace_score(moments)
        score.backward()
        self.assertTrue(torch.isfinite(f.grad).all())
        self.assertTrue(torch.isfinite(g.grad).all())

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

    def test_relative_ridge_is_invariant_across_small_and_large_feature_scales(self) -> None:
        generator = torch.Generator().manual_seed(17)
        f = 3.0 * torch.randn(2048, 3, generator=generator, dtype=torch.float64)
        noise = torch.randn(2048, 4, 3, generator=generator, dtype=torch.float64)
        g = 0.6 * f.unsqueeze(1) + noise
        spectra = []
        traces = []
        logdets = []
        for scale in (1e-3, 1.0, 1e3):
            moments = estimate_moments(scale * f, scale * g)
            spectra.append(torch.linalg.svdvals(whitened_cross_operator(moments, ridge=1e-3)))
            traces.append(trace_score(moments, ridge=1e-3))
            logdets.append(logdet_score(moments, ridge=1e-3))
        for index in (1, 2):
            self.assertTrue(torch.allclose(spectra[0], spectra[index], atol=1e-10, rtol=1e-10))
            self.assertTrue(torch.allclose(traces[0], traces[index], atol=1e-10, rtol=1e-10))
            self.assertTrue(torch.allclose(logdets[0], logdets[index], atol=1e-10, rtol=1e-10))

    def test_zero_variance_is_finite_and_nonfinite_covariance_is_explicit(self) -> None:
        f = torch.zeros(32, 3, dtype=torch.float64)
        g = torch.zeros(32, 2, 3, dtype=torch.float64)
        moments = estimate_moments(f, g)
        operator = whitened_cross_operator(moments, ridge=1e-3)
        calibration = fit_spectral_calibration(f, g, ridge=1e-3)
        values = torch.cat((
            operator.flatten(),
            calibration.singular_values,
            calibration.eigenvalues,
            trace_score(moments, ridge=1e-3).reshape(1),
            logdet_score(moments, ridge=1e-3).reshape(1),
        ))
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.equal(operator, torch.zeros_like(operator)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            relative_ridge_scale(torch.tensor([[float("nan")]], dtype=torch.float64))

    def test_heldout_full_svd_is_rotation_invariant_but_diagonal_is_not(self) -> None:
        generator = torch.Generator().manual_seed(23)
        samples = 4096
        f = torch.randn(samples, 3, generator=generator, dtype=torch.float64)
        noise = 0.05 * torch.randn(samples, 2, 3, generator=generator, dtype=torch.float64)
        g = f[:, None, :] * torch.tensor([0.9, 0.6, 0.2], dtype=torch.float64) + noise
        calibration = {
            "mean_f": torch.zeros(1, 3, dtype=torch.float64),
            "mean_g": torch.zeros(1, 3, dtype=torch.float64),
            "transform_f": torch.eye(3, dtype=torch.float64),
            "transform_g": torch.eye(3, dtype=torch.float64),
        }
        angle = torch.tensor(0.71, dtype=torch.float64)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        zero, one = torch.zeros_like(angle), torch.ones_like(angle)
        rotation = torch.stack((
            torch.stack((cosine, -sine, zero)),
            torch.stack((sine, cosine, zero)),
            torch.stack((zero, zero, one)),
        ))
        base = evaluate_heldout_spectrum(f, g, calibration)
        rotated = evaluate_heldout_spectrum(f @ rotation, g, calibration)
        self.assertFalse(torch.allclose(base.diagonal_correlations, rotated.diagonal_correlations))
        self.assertTrue(torch.allclose(base.singular_values, rotated.singular_values, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(base.eigenvalues, rotated.eigenvalues, atol=1e-12, rtol=1e-12))

    def test_logdet_clipping_does_not_modify_raw_eigenvalues(self) -> None:
        raw = torch.tensor([1.2, 0.5, 0.1], dtype=torch.float64)
        preserved = raw.clone()
        score, clipped = clipped_logdet_from_eigenvalues(raw, margin=1e-7)
        self.assertTrue(torch.isfinite(score))
        self.assertEqual(clipped, 1)
        self.assertTrue(torch.equal(raw, preserved))


if __name__ == "__main__":
    unittest.main()
