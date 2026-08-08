import unittest

import torch

from fmca_av.analytic import finite_channel_spectrum
from fmca_av.data.gaussian import gaussian_eigenvalues


class AnalyticReferenceTests(unittest.TestCase):
    def test_finite_independent_channel_has_only_zero_nonconstant_modes(self) -> None:
        p_x = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        p_y = torch.tensor([0.4, 0.6], dtype=torch.float64)
        spectrum = finite_channel_spectrum(p_x[:, None] * p_y[None, :])
        self.assertAlmostEqual(float(spectrum.singular_values_with_constant[0]), 1.0)
        self.assertTrue(torch.all(spectrum.eigenvalues < 1e-28))

    def test_gaussian_constant_mode_is_omitted(self) -> None:
        eigenvalues = gaussian_eigenvalues(noise_variance=1.0, count=4)
        expected = torch.tensor([0.5, 0.25, 0.125, 0.0625], dtype=torch.float64)
        self.assertTrue(torch.equal(eigenvalues, expected))


if __name__ == "__main__":
    unittest.main()

