import unittest

import torch

from scripts.run_e1_neural_nonlinear import clean_samples, numerical_oracle, parse_conditions


class NeuralNonlinearToyTests(unittest.TestCase):
    def test_condition_parser_rejects_duplicates(self) -> None:
        self.assertEqual(parse_conditions("two_moons:1,gmm:2,spiral:3"), [
            ("two_moons", 1), ("gmm", 2), ("spiral", 3),
        ])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_conditions("gmm:1,gmm:1")

    def test_all_families_produce_finite_continuous_parents(self) -> None:
        for index, family in enumerate(("two_moons", "gmm", "spiral")):
            values = clean_samples(family, 128, torch.Generator().manual_seed(100 + index))
            self.assertEqual(values.shape, (128, 2))
            self.assertTrue(torch.isfinite(values).all())
            self.assertGreater(float(values.std()), 0.0)

    def test_numerical_oracle_has_valid_nonconstant_spectrum(self) -> None:
        parents = clean_samples("two_moons", 512, torch.Generator().manual_seed(3))
        views = parents[:, None, :] + 0.2 * torch.randn(
            512, 4, 2, generator=torch.Generator().manual_seed(4),
        )
        eigenvalues = numerical_oracle(parents, views, bins=5)
        self.assertTrue(torch.isfinite(eigenvalues).all())
        self.assertTrue((eigenvalues >= 0).all())
        self.assertLessEqual(float(eigenvalues.max()), 1.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
