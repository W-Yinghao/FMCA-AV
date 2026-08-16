"""Stage-B/C machinery: coordinate algebra, split disjointness, cross-fit.

These tests pin the mutations the audit found uncovered: whitener applied
before mean subtraction, silently-disabled centering, dtype downcasting of
frozen coordinates, and the (previously untested) split/cross-fit path.
"""

import unittest

import torch

from fmca_av.certificate.coordinates import fit_level_coordinates
from fmca_av.certificate.counterexamples import case_closed_chain
from fmca_av.certificate.estimation import (
    ChainFeatureBatch,
    crossfit_edge_and_endpoint,
    encode_chain_batch,
    level_calibration_features,
    split_chain_batch,
)


class CoordinateAlgebraTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(2)
        # Deliberately shifted, scaled, and correlated raw features.
        base = torch.randn(4000, 3, generator=generator, dtype=torch.float64)
        mixing = torch.tensor(
            [[1.5, 0.3, 0.0], [0.0, 0.8, 0.4], [0.0, 0.0, 2.0]], dtype=torch.float64
        )
        self.features = base @ mixing + torch.tensor([5.0, -3.0, 0.7], dtype=torch.float64)
        self.coordinates = fit_level_coordinates(self.features, ridge=1e-6)

    def test_encode_is_center_then_whiten(self) -> None:
        expected = (self.features - self.coordinates.mean) @ self.coordinates.whitener
        self.assertTrue(torch.allclose(self.coordinates.encode(self.features), expected))
        # The wrong order (whiten then subtract the raw mean) must differ.
        wrong = self.features @ self.coordinates.whitener - self.coordinates.mean
        self.assertGreater(float((expected - wrong).abs().max()), 0.1)

    def test_encoded_calibration_data_is_centered_and_white(self) -> None:
        encoded = self.coordinates.encode(self.features)
        self.assertLess(float(encoded.mean(dim=0).abs().max()), 1e-8)
        gram = encoded.transpose(0, 1) @ encoded / encoded.shape[0]
        self.assertTrue(
            torch.allclose(gram, torch.eye(3, dtype=torch.float64), atol=1e-4)
        )

    def test_low_precision_features_are_promoted_not_coordinates_downcast(self) -> None:
        half = self.features.to(torch.float16)
        encoded = self.coordinates.encode(half)
        self.assertEqual(encoded.dtype, torch.float64)
        reference = self.coordinates.encode(self.features)
        self.assertLess(float((encoded - reference).abs().max()), 0.05)

    def test_non_finite_features_fail_loud(self) -> None:
        poisoned = self.features.clone()
        poisoned[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.coordinates.encode(poisoned)


class SplitAndCrossfitTests(unittest.TestCase):
    def setUp(self) -> None:
        case = case_closed_chain()
        self.case = case
        calibration = case.sample(2000, generator=torch.Generator().manual_seed(51))
        self.coordinates = [
            fit_level_coordinates(level_calibration_features(calibration, level))
            for level in range(calibration.num_levels)
        ]
        raw = case.sample(8000, generator=torch.Generator().manual_seed(52))
        self.encoded = encode_chain_batch(raw, self.coordinates)

    def test_split_is_disjoint_and_exhaustive_over_parents(self) -> None:
        # Tag each parent with a unique value in an extra pseudo-feature to
        # track identity through the split: use the root feature row itself.
        parts = split_chain_batch(self.encoded, [0.3, 0.3, 0.4], torch.Generator().manual_seed(53))
        self.assertEqual(sum(part.num_parents for part in parts), self.encoded.num_parents)
        rows = torch.cat([part.chain[0] for part in parts], dim=0)
        original = self.encoded.chain[0]
        self.assertTrue(
            torch.allclose(
                rows.sort(dim=0).values, original.sort(dim=0).values, atol=1e-12
            )
        )

    def test_split_rejects_bad_fractions(self) -> None:
        with self.assertRaises(ValueError):
            split_chain_batch(self.encoded, [0.7, 0.7])
        with self.assertRaises(ValueError):
            split_chain_batch(self.encoded, [1.0])

    def test_crossfit_folds_recover_population_operators(self) -> None:
        folds = crossfit_edge_and_endpoint(self.encoded, torch.Generator().manual_seed(54))
        self.assertEqual(len(folds), 2)
        for edges, c_dir in folds:
            for edge, target in zip(edges, self.case.population_edges):
                self.assertTrue(torch.allclose(edge, target, atol=0.08))
            self.assertTrue(torch.allclose(c_dir, self.case.population_direct, atol=0.08))

    def test_zero_view_children_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ChainFeatureBatch(
                chain=[torch.zeros(4, 2), torch.zeros(4, 2)],
                children=[torch.zeros(4, 0, 2)],
            )


if __name__ == "__main__":
    unittest.main()
