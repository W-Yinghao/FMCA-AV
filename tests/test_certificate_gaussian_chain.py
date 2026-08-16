"""AR(1)-Hermite analytic reference: exact closure without truncation,
closed-form projection defect with interior truncation, and empirical
convergence of the full pipeline."""

import unittest

import torch

from fmca_av.certificate.coordinates import fit_level_coordinates
from fmca_av.certificate.estimation import (
    encode_chain_batch,
    estimate_edge_operators,
    estimate_endpoint_operator,
    level_calibration_features,
)
from fmca_av.certificate.gaussian_chain import (
    GaussianHermiteChain,
    normalized_hermite_features,
)
from fmca_av.certificate.triplet import certificate_report, compose_edge_operators


class PopulationTests(unittest.TestCase):
    def test_full_interfaces_close_exactly(self) -> None:
        chain = GaussianHermiteChain(rhos=[0.8, 0.6], level_orders=[[1, 2, 3]] * 3)
        composed = compose_edge_operators(chain.population_edges())
        self.assertTrue(torch.allclose(composed, chain.population_direct(), atol=1e-12))
        self.assertTrue(torch.allclose(composed, chain.population_composed(), atol=1e-12))
        self.assertAlmostEqual(float(chain.population_defect().abs().max()), 0.0)

    def test_interior_truncation_produces_the_analytic_defect(self) -> None:
        chain = GaussianHermiteChain(
            rhos=[0.8, 0.6],
            level_orders=[[1, 2, 3], [1, 2], [1, 2, 3]],
        )
        composed = compose_edge_operators(chain.population_edges())
        self.assertTrue(torch.allclose(composed, chain.population_composed(), atol=1e-12))
        defect = chain.population_direct() - composed
        # Only mode 3 is dropped by the interior interface.
        expected = torch.zeros(3, 3, dtype=torch.float64)
        expected[2, 2] = (0.8 * 0.6) ** 3
        self.assertTrue(torch.allclose(defect, expected, atol=1e-12))
        report = certificate_report(chain.population_direct(), edges=chain.population_edges())
        self.assertAlmostEqual(report.delta_operator, (0.8 * 0.6) ** 3, places=10)

    def test_hermite_features_are_orthonormal_in_the_sample_limit(self) -> None:
        generator = torch.Generator().manual_seed(3)
        samples = torch.randn(200000, generator=generator, dtype=torch.float64)
        features = normalized_hermite_features(samples, [1, 2, 3, 4])
        gram = features.transpose(0, 1) @ features / features.shape[0]
        self.assertTrue(torch.allclose(gram, torch.eye(4, dtype=torch.float64), atol=0.08))
        self.assertLess(float(features.mean(dim=0).abs().max()), 0.05)


class EmpiricalTests(unittest.TestCase):
    def _pipeline(self, chain: GaussianHermiteChain, parents: int, seed: int):
        calibration = chain.sample(
            max(parents // 4, 1000), generator=torch.Generator().manual_seed(seed)
        )
        coordinates = [
            fit_level_coordinates(level_calibration_features(calibration, level))
            for level in range(calibration.num_levels)
        ]
        batch = chain.sample(parents, generator=torch.Generator().manual_seed(seed + 1))
        encoded = encode_chain_batch(batch, coordinates)
        edges = estimate_edge_operators(encoded)
        c_dir = estimate_endpoint_operator(encoded)
        return edges, c_dir

    def test_empirical_pipeline_recovers_population_operators(self) -> None:
        chain = GaussianHermiteChain(rhos=[0.8, 0.6], level_orders=[[1, 2, 3]] * 3)
        edges, c_dir = self._pipeline(chain, parents=20000, seed=13)
        for edge, target in zip(edges, chain.population_edges()):
            self.assertTrue(torch.allclose(edge, target, atol=0.06), msg=str(edge))
        self.assertTrue(torch.allclose(c_dir, chain.population_direct(), atol=0.06))
        report = certificate_report(c_dir, edges=edges)
        count = report.certified_spectrum.shape[0]
        self.assertTrue(
            bool(
                (
                    report.certified_spectrum
                    <= report.endpoint_singular_values[:count] + 0.03
                ).all()
            )
        )

    def test_empirical_defect_matches_the_analytic_truncation_defect(self) -> None:
        chain = GaussianHermiteChain(
            rhos=[0.8, 0.6], level_orders=[[1, 2, 3], [1, 2], [1, 2, 3]]
        )
        edges, c_dir = self._pipeline(chain, parents=20000, seed=17)
        report = certificate_report(c_dir, edges=edges)
        self.assertAlmostEqual(report.delta_operator, (0.8 * 0.6) ** 3, delta=0.05)

    def test_operator_error_shrinks_with_sample_size(self) -> None:
        chain = GaussianHermiteChain(rhos=[0.8, 0.6], level_orders=[[1, 2, 3]] * 3)
        errors = {}
        for parents in (1000, 16000):
            edges, c_dir = self._pipeline(chain, parents=parents, seed=19)
            error = float((c_dir - chain.population_direct()).abs().max())
            for edge, target in zip(edges, chain.population_edges()):
                error = max(error, float((edge - target).abs().max()))
            errors[parents] = error
        self.assertLess(errors[16000], errors[1000] * 0.8)


if __name__ == "__main__":
    unittest.main()
