"""Control battery: positive controls must be invariant, negative controls
must move.  Every check runs on analytic or seeded synthetic chains."""

import unittest

import torch

from fmca_av.certificate.controls import (
    edgewise_calibrated_edges,
    endpoint_pairing_shuffle,
    pairing_noise_floor,
    random_orthogonal,
    rotate_interface,
    shuffle_edge_order,
    uncentered_coordinates,
)
from fmca_av.certificate.coordinates import fit_level_coordinates
from fmca_av.certificate.counterexamples import case_closed_chain, case_isospectral_mismatch
from fmca_av.certificate.estimation import (
    ChainFeatureBatch,
    encode_chain_batch,
    estimate_edge_operators,
    estimate_endpoint_operator,
    level_calibration_features,
)
from fmca_av.certificate.triplet import (
    certificate_report,
    compose_edge_operators,
    naive_singular_value_product,
)


class GaugeControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.edges = case_isospectral_mismatch().population_edges
        self.c_dir = case_isospectral_mismatch().population_direct
        self.rotation = random_orthogonal(2, torch.Generator().manual_seed(5))

    def test_two_sided_rotation_leaves_composition_invariant(self) -> None:
        rotated = rotate_interface(self.edges, interface=1, rotation=self.rotation, sides="both")
        original = compose_edge_operators(self.edges)
        invariant = compose_edge_operators(rotated)
        self.assertTrue(torch.allclose(original, invariant, atol=1e-12))

    def test_two_sided_rotation_leaves_report_invariant(self) -> None:
        rotated = rotate_interface(self.edges, interface=1, rotation=self.rotation, sides="both")
        base = certificate_report(self.c_dir, edges=self.edges)
        turned = certificate_report(self.c_dir, edges=rotated)
        self.assertAlmostEqual(base.delta_operator, turned.delta_operator, places=10)
        self.assertAlmostEqual(base.delta_frobenius, turned.delta_frobenius, places=10)
        self.assertTrue(
            torch.allclose(base.certified_spectrum, turned.certified_spectrum, atol=1e-10)
        )

    def test_one_sided_rotation_changes_the_composition(self) -> None:
        for sides in ("left", "right"):
            rotated = rotate_interface(self.edges, interface=1, rotation=self.rotation, sides=sides)
            difference = compose_edge_operators(rotated) - compose_edge_operators(self.edges)
            self.assertGreater(float(difference.abs().max()), 0.01, msg=sides)


class OrderingAndNaiveProductTests(unittest.TestCase):
    def test_layer_order_shuffle_changes_heterogeneous_composition(self) -> None:
        generator = torch.Generator().manual_seed(2)
        edges = [
            torch.randn(3, 3, generator=generator, dtype=torch.float64),
            torch.randn(3, 3, generator=generator, dtype=torch.float64),
        ]
        shuffled = shuffle_edge_order(edges, [1, 0])
        difference = compose_edge_operators(shuffled) - compose_edge_operators(edges)
        self.assertGreater(float(difference.abs().max()), 0.1)

    def test_naive_singular_value_product_fails_on_misaligned_modes(self) -> None:
        edges = [
            torch.diag(torch.tensor([0.9, 0.4], dtype=torch.float64)),
            torch.diag(torch.tensor([0.4, 0.9], dtype=torch.float64)),
        ]
        true_spectrum = torch.linalg.svdvals(compose_edge_operators(edges))
        naive = naive_singular_value_product(edges)
        self.assertTrue(torch.allclose(true_spectrum, torch.tensor([0.36, 0.36], dtype=torch.float64)))
        self.assertGreater(float((naive - true_spectrum).abs().max()), 0.2)

    def test_naive_product_stays_wrong_under_interface_rotation(self) -> None:
        edges = [
            torch.diag(torch.tensor([0.9, 0.4], dtype=torch.float64)),
            torch.diag(torch.tensor([0.4, 0.9], dtype=torch.float64)),
        ]
        rotation = random_orthogonal(2, torch.Generator().manual_seed(9))
        rotated = rotate_interface(edges, interface=1, rotation=rotation, sides="both")
        true_spectrum = torch.linalg.svdvals(compose_edge_operators(rotated))
        naive = naive_singular_value_product(rotated)
        self.assertTrue(torch.allclose(true_spectrum, torch.tensor([0.36, 0.36], dtype=torch.float64), atol=1e-10))
        self.assertGreater(float((naive - true_spectrum).abs().max()), 0.2)


class PairingShuffleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = case_closed_chain()
        generator = torch.Generator().manual_seed(21)
        calibration = self.case.sample(2000, generator=generator)
        coordinates = [
            fit_level_coordinates(level_calibration_features(calibration, level))
            for level in range(calibration.num_levels)
        ]
        batch = self.case.sample(5000, generator=torch.Generator().manual_seed(22))
        self.encoded = encode_chain_batch(batch, coordinates)

    def test_parent_pairing_shuffle_sits_far_below_the_true_edge_norm(self) -> None:
        true_norm = float(
            torch.linalg.matrix_norm(estimate_edge_operators(self.encoded)[0], ord="fro")
        )
        floor = pairing_noise_floor(
            self.encoded.chain[0],
            self.encoded.children[0],
            repeats=10,
            generator=torch.Generator().manual_seed(23),
        )
        self.assertGreater(true_norm, 0.5)
        self.assertLess(float(floor.max()), 0.25 * true_norm)

    def test_endpoint_pairing_shuffle_kills_the_endpoint_operator(self) -> None:
        true_norm = float(
            torch.linalg.matrix_norm(estimate_endpoint_operator(self.encoded), ord="fro")
        )
        shuffled = endpoint_pairing_shuffle(self.encoded, torch.Generator().manual_seed(24))
        self.assertGreater(true_norm, 0.4)
        self.assertLess(float(torch.linalg.matrix_norm(shuffled, ord="fro")), 0.25 * true_norm)


class CenteringControlTests(unittest.TestCase):
    def test_missing_centering_manufactures_a_unit_mode(self) -> None:
        case = case_closed_chain()
        generator = torch.Generator().manual_seed(31)
        calibration = case.sample(4000, generator=generator)
        batch = case.sample(8000, generator=torch.Generator().manual_seed(32))
        offset = 3.0

        def shift(chain_batch: ChainFeatureBatch) -> ChainFeatureBatch:
            return ChainFeatureBatch(
                chain=[states + offset for states in chain_batch.chain],
                children=[descendants + offset for descendants in chain_batch.children],
                endpoint_descendants=chain_batch.endpoint_descendants + offset,
            )

        shifted_calibration, shifted_batch = shift(calibration), shift(batch)
        centered = [
            fit_level_coordinates(level_calibration_features(shifted_calibration, level))
            for level in range(shifted_calibration.num_levels)
        ]
        uncentered = uncentered_coordinates(shifted_calibration)
        centered_comp = compose_edge_operators(
            estimate_edge_operators(encode_chain_batch(shifted_batch, centered))
        )
        uncentered_comp = compose_edge_operators(
            estimate_edge_operators(encode_chain_batch(shifted_batch, uncentered))
        )
        top_centered = float(torch.linalg.matrix_norm(centered_comp, ord=2))
        top_uncentered = float(torch.linalg.matrix_norm(uncentered_comp, ord=2))
        # The constant mode masquerades as near-perfect transmission.
        self.assertGreater(top_uncentered, 0.95)
        self.assertLess(top_centered, 0.85)


class EdgewiseCalibrationControlTests(unittest.TestCase):
    def test_edgewise_coordinates_break_the_shared_interface(self) -> None:
        case = case_closed_chain()
        calibration = case.sample(300, generator=torch.Generator().manual_seed(41))
        batch = case.sample(5000, generator=torch.Generator().manual_seed(42))
        shared_coordinates = [
            fit_level_coordinates(level_calibration_features(calibration, level))
            for level in range(calibration.num_levels)
        ]
        shared = estimate_edge_operators(encode_chain_batch(batch, shared_coordinates))
        edgewise = edgewise_calibrated_edges(batch, calibration)
        # The two whitenings of the shared interior level disagree, so the
        # composed operator drifts from the shared-coordinate composition.
        difference = compose_edge_operators(edgewise) - compose_edge_operators(shared)
        self.assertGreater(float(difference.abs().max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
