"""Wave 0 acceptance: the six frozen analytic cases.

Population values are asserted against the frozen table (2026-08-15), and
the FULL empirical pipeline (Stage-B calibration on a disjoint sample,
Stage-C estimation, certificate report) must recover them.  The certificate
may accept ONLY the closed-chain positive case.
"""

import math
import unittest

import torch

from fmca_av.certificate.coordinates import fit_level_coordinates
from fmca_av.certificate.counterexamples import all_counterexamples, case_closed_chain
from fmca_av.certificate.estimation import (
    encode_chain_batch,
    estimate_edge_operators,
    estimate_endpoint_operator,
    level_calibration_features,
)
from fmca_av.certificate.triplet import certificate_report, compose_edge_operators

CERTIFY_THRESHOLD = 0.05


def _fit_coordinates(case, parents, seed, ridge=1e-3):
    generator = torch.Generator().manual_seed(seed)
    calibration = case.sample(parents, children_per_edge=4, endpoint_descendants=4, generator=generator)
    return [
        fit_level_coordinates(level_calibration_features(calibration, level), ridge=ridge)
        for level in range(calibration.num_levels)
    ]


def _empirical_report(case, parents, seed, top_k=4):
    coordinates = _fit_coordinates(case, max(parents // 4, 500), seed)
    generator = torch.Generator().manual_seed(seed + 1)
    batch = case.sample(parents, children_per_edge=4, endpoint_descendants=4, generator=generator)
    encoded = encode_chain_batch(batch, coordinates)
    edges = estimate_edge_operators(encoded)
    c_dir = estimate_endpoint_operator(encoded)
    return certificate_report(c_dir, edges=edges, top_k=top_k), edges, c_dir


class PopulationTableTests(unittest.TestCase):
    def test_nilpotent_edges_have_unit_mass_but_zero_product(self) -> None:
        case = next(c for c in all_counterexamples() if c.name == "nilpotent_interface")
        report = case.population_report()
        self.assertAlmostEqual(report.edge_frobenius_norms[0], 1.0)
        self.assertAlmostEqual(report.edge_frobenius_norms[1], 1.0)
        self.assertAlmostEqual(float(report.path_singular_values.max()), 0.0)
        self.assertAlmostEqual(float(report.endpoint_singular_values.max()), 0.0)

    def test_hallucinated_path_fabricates_half(self) -> None:
        case = next(c for c in all_counterexamples() if c.name == "hallucinated_path")
        self.assertAlmostEqual(float(case.population_edges[0]), 1.0 / math.sqrt(2.0))
        self.assertAlmostEqual(float(case.population_edges[1]), 1.0 / math.sqrt(2.0))
        report = case.population_report()
        self.assertAlmostEqual(float(report.path_singular_values[0]), 0.5)
        self.assertAlmostEqual(float(report.endpoint_singular_values[0]), 0.0)
        self.assertAlmostEqual(report.delta_operator, 0.5)
        self.assertAlmostEqual(float(report.certified_spectrum[0]), 0.0)

    def test_leaky_interface_hides_perfect_endpoint(self) -> None:
        case = next(c for c in all_counterexamples() if c.name == "leaky_interface")
        report = case.population_report()
        self.assertAlmostEqual(float(report.path_singular_values[0]), 0.0)
        self.assertAlmostEqual(float(report.endpoint_singular_values[0]), 1.0)
        self.assertAlmostEqual(float(report.certified_spectrum[0]), 0.0)

    def test_zero_operator_closure_is_perfect_but_certifies_nothing(self) -> None:
        case = next(c for c in all_counterexamples() if c.name == "zero_operator")
        report = case.population_report()
        self.assertAlmostEqual(report.delta_frobenius, 0.0)
        self.assertAlmostEqual(float(report.certified_spectrum.max()), 0.0)

    def test_isospectral_pair_needs_the_full_matrix(self) -> None:
        case = next(c for c in all_counterexamples() if c.name == "isospectral_mismatch")
        report = case.population_report()
        self.assertTrue(
            torch.allclose(report.endpoint_singular_values, report.path_singular_values, atol=1e-12)
        )
        self.assertAlmostEqual(report.alignment, 0.0)
        self.assertAlmostEqual(report.delta_operator, 0.5)
        self.assertAlmostEqual(report.delta_frobenius, math.sqrt(0.5))
        self.assertAlmostEqual(float(report.certified_spectrum[0]), 0.0)

    def test_closed_chain_composes_exactly_and_certifies(self) -> None:
        case = case_closed_chain()
        composed = compose_edge_operators(case.population_edges)
        self.assertTrue(torch.allclose(composed, case.population_direct, atol=1e-12))
        report = case.population_report()
        self.assertAlmostEqual(report.delta_frobenius, 0.0)
        self.assertAlmostEqual(float(report.certified_spectrum[0]), 0.9 * 0.8)
        self.assertGreater(float(report.certified_spectrum[0]), CERTIFY_THRESHOLD)

    def test_certificate_accepts_only_the_positive_case(self) -> None:
        for case in all_counterexamples():
            report = case.population_report()
            accepted = float(report.certified_spectrum.max()) > CERTIFY_THRESHOLD
            self.assertEqual(accepted, case.certifies, msg=case.name)

    def test_polarization_identity_holds_for_every_case(self) -> None:
        for case in all_counterexamples():
            report = case.population_report()
            self.assertLess(report.polarization_residual, 1e-10, msg=case.name)

    def test_weyl_bound_holds_for_every_case(self) -> None:
        for case in all_counterexamples():
            report = case.population_report()
            count = min(report.certified_spectrum.shape[0], report.endpoint_singular_values.shape[0])
            self.assertTrue(
                bool(
                    (
                        report.certified_spectrum[:count]
                        <= report.endpoint_singular_values[:count] + 1e-12
                    ).all()
                ),
                msg=case.name,
            )


class EmpiricalPipelineTests(unittest.TestCase):
    def test_empirical_reports_match_population(self) -> None:
        for case in all_counterexamples():
            population = case.population_report()
            empirical, edges, c_dir = _empirical_report(case, parents=20000, seed=7)
            for edge_index, edge in enumerate(edges):
                self.assertTrue(
                    torch.allclose(edge, case.population_edges[edge_index], atol=0.05),
                    msg=f"{case.name} edge {edge_index}: {edge} vs {case.population_edges[edge_index]}",
                )
            self.assertTrue(
                torch.allclose(c_dir, case.population_direct, atol=0.05),
                msg=f"{case.name} direct: {c_dir}",
            )
            self.assertLess(abs(empirical.delta_operator - population.delta_operator), 0.06, msg=case.name)
            accepted = float(empirical.certified_spectrum.max()) > CERTIFY_THRESHOLD
            self.assertEqual(accepted, case.certifies, msg=case.name)

    def test_estimation_error_decreases_with_sample_size(self) -> None:
        case = case_closed_chain()
        errors = {}
        for parents in (1000, 16000):
            _, edges, c_dir = _empirical_report(case, parents=parents, seed=11)
            error = max(
                float((edge - target).abs().max())
                for edge, target in zip(edges, case.population_edges)
            )
            error = max(error, float((c_dir - case.population_direct).abs().max()))
            errors[parents] = error
        self.assertLess(errors[16000], errors[1000] * 0.8)

    def test_more_children_reduce_edge_estimator_error(self) -> None:
        case = case_closed_chain()
        coordinates = _fit_coordinates(case, 2000, seed=3)
        target = case.population_edges[0]
        mean_errors = {}
        for children in (1, 16):
            repeats = []
            for repeat in range(12):
                generator = torch.Generator().manual_seed(1000 + repeat)
                batch = case.sample(600, children_per_edge=children, endpoint_descendants=1, generator=generator)
                encoded = encode_chain_batch(batch, coordinates)
                edge = estimate_edge_operators(encoded)[0]
                repeats.append(float((edge - target).square().sum()))
            mean_errors[children] = sum(repeats) / len(repeats)
        self.assertLess(mean_errors[16], mean_errors[1])


if __name__ == "__main__":
    unittest.main()
