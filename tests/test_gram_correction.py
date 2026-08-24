"""Step-1 acceptance tests for the Gram correction (frozen appendum §6).

Each test also pins the failure it guards against: the corrected and
uncorrected quantities are computed side by side, so a regression that
silently drops a Gram inverse turns the assertion around rather than
leaving it vacuously true.
"""

import unittest

import torch

from fmca_av.certificate.finite_sample import (
    matrix_bernstein_radius,
    path_radius,
    tiered_certificate,
)
from fmca_av.certificate.gram import (
    build_correction,
    corrected_composition,
    corrected_endpoint,
    cumulative_interface_attribution,
    gram_matrix,
    spectral_pseudo_inverse,
)
from fmca_av.certificate.triplet import compose_edge_operators


def _orthonormal(dim, seed):
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float64))
    return q


def _chain(dim=6, seed=0):
    """A population chain in orthonormal coordinates: Gram is the identity."""

    generator = torch.Generator().manual_seed(seed)
    edges = [0.6 * _orthonormal(dim, seed + 1) @ torch.diag(
                 torch.linspace(0.9, 0.2, dim, dtype=torch.float64)),
             0.7 * _orthonormal(dim, seed + 2) @ torch.diag(
                 torch.linspace(0.8, 0.3, dim, dtype=torch.float64))]
    c_dir = compose_edge_operators(edges) + 0.05 * torch.randn(
        dim, dim, generator=generator, dtype=torch.float64)
    return edges, c_dir


class GramIsANoOpWhenOrthonormal(unittest.TestCase):
    """Counterexample regression: G = I must leave the triplet untouched."""

    def test_identity_gram_changes_nothing(self):
        edges, c_dir = _chain()
        dim = edges[0].shape[0]
        grams = [torch.eye(dim, dtype=torch.float64) for _ in range(3)]
        correction = build_correction(grams)
        torch.testing.assert_close(corrected_composition(edges, correction),
                                   compose_edge_operators(edges))
        torch.testing.assert_close(corrected_endpoint(c_dir, correction), c_dir)
        self.assertEqual(correction.retained_ranks, [dim] * 3)
        self.assertLess(max(correction.metric_deviation), 1e-12)


class InvertibleReparameterizationTest(unittest.TestCase):
    """§4.2: the corrected triplet is invariant, the surrogate is not.

    Reparameterizing level l by an invertible A_l sends B_{l,l+1} to
    A_l^T B A_{l+1} and G_l to A_l^T G_l A_l, so the corrected product
    -- which depends only on the subspace -- must not move.
    """

    def _reparameterized(self, edges, c_dir, maps):
        moved_edges = [maps[i].transpose(0, 1) @ edges[i] @ maps[i + 1]
                       for i in range(len(edges))]
        moved_dir = maps[0].transpose(0, 1) @ c_dir @ maps[-1]
        grams = [m.transpose(0, 1) @ m for m in maps]
        return moved_edges, moved_dir, grams

    def test_corrected_defect_is_invariant_and_surrogate_is_not(self):
        edges, c_dir = _chain(seed=3)
        dim = edges[0].shape[0]
        generator = torch.Generator().manual_seed(11)
        maps = []
        for level in range(3):
            base = _orthonormal(dim, 20 + level)
            scale = torch.diag(torch.linspace(0.6, 1.7, dim, dtype=torch.float64))
            maps.append(base @ scale)

        moved_edges, moved_dir, grams = self._reparameterized(edges, c_dir, maps)
        correction = build_correction(grams)

        reference = float(torch.linalg.matrix_norm(
            c_dir - compose_edge_operators(edges), ord="fro"))
        corrected = float(torch.linalg.matrix_norm(
            corrected_endpoint(moved_dir, correction)
            - corrected_composition(moved_edges, correction), ord="fro"))
        surrogate = float(torch.linalg.matrix_norm(
            moved_dir - compose_edge_operators(moved_edges), ord="fro"))

        # Corrected recovers the orthonormal-coordinate answer up to the
        # unitary freedom in G^{-1/2}; the Frobenius norm is invariant.
        self.assertAlmostEqual(corrected, reference, places=8)
        # The mutation this pins: dropping the correction moves the number.
        self.assertGreater(abs(surrogate - reference), 1e-3)

    def test_orthogonal_maps_leave_even_the_surrogate_alone(self):
        """Two-sided orthogonal gauge invariance survives, as before."""

        edges, c_dir = _chain(seed=5)
        dim = edges[0].shape[0]
        maps = [_orthonormal(dim, 40 + level) for level in range(3)]
        moved_edges, moved_dir, grams = self._reparameterized(edges, c_dir, maps)
        reference = float(torch.linalg.matrix_norm(
            c_dir - compose_edge_operators(edges), ord="fro"))
        surrogate = float(torch.linalg.matrix_norm(
            moved_dir - compose_edge_operators(moved_edges), ord="fro"))
        self.assertAlmostEqual(surrogate, reference, places=10)


class RidgeSweepTest(unittest.TestCase):
    """The surrogate moves with the ridge; the corrected quantity does not."""

    def _sampled_gram(self, ridge, dim=6, samples=4000, seed=7):
        generator = torch.Generator().manual_seed(seed)
        basis = _orthonormal(dim, 99)
        spectrum = torch.linspace(1.0, 0.02, dim, dtype=torch.float64)
        raw = torch.randn(samples, dim, generator=generator, dtype=torch.float64) * spectrum.sqrt()
        raw = raw @ basis.transpose(0, 1)
        covariance = raw.transpose(0, 1) @ raw / samples
        scale = float(covariance.diagonal().mean())
        values, vectors = torch.linalg.eigh(covariance + ridge * scale * torch.eye(dim, dtype=torch.float64))
        whitener = (vectors * values.clamp_min(1e-12).rsqrt()) @ vectors.transpose(0, 1)
        return gram_matrix(raw @ whitener)

    def test_corrected_recovers_the_orthonormal_answer_at_every_ridge(self):
        """The sharp form: B = F^T T F with F^T F = G means the ridge
        coordinates carry B_ridge = G^{1/2} B G^{1/2}.  Inserting the Gram
        inverses must then return the orthonormal-coordinate defect
        exactly, at any ridge, while the surrogate drifts with it."""

        edges, c_dir = _chain(seed=13)
        reference = float(torch.linalg.matrix_norm(
            c_dir - compose_edge_operators(edges), ord="fro"))
        corrected_values, surrogate_values = [], []
        for ridge in (1e-3, 1e-2, 1e-1):
            gram = self._sampled_gram(ridge)
            root, _ = spectral_pseudo_inverse(gram, 0.5)
            correction = build_correction([gram, gram, gram])
            moved_edges = [root @ edge @ root for edge in edges]
            moved_dir = root @ c_dir @ root
            corrected_values.append(float(torch.linalg.matrix_norm(
                corrected_endpoint(moved_dir, correction)
                - corrected_composition(moved_edges, correction), ord="fro")))
            surrogate_values.append(float(torch.linalg.matrix_norm(
                moved_dir - compose_edge_operators(moved_edges), ord="fro")))
        for value in corrected_values:
            self.assertAlmostEqual(value, reference, places=6)
        # The mutation this pins: without the correction the number moves
        # with the ridge.  Direction is deliberately NOT asserted -- the
        # same lesson that retired the one-sided-rotation criterion applies
        # here, the drift sign depends on the operators, not on the ridge.
        corrected_spread = max(corrected_values) - min(corrected_values)
        surrogate_spread = max(surrogate_values) - min(surrogate_values)
        self.assertGreater(surrogate_spread, 100 * max(corrected_spread, 1e-12))
        for value in surrogate_values:
            self.assertGreater(abs(value - reference), 1e-3)

    def test_ridge_pushes_the_gram_away_from_identity(self):
        deviations = [float(torch.linalg.matrix_norm(
            self._sampled_gram(ridge) - torch.eye(6, dtype=torch.float64), ord=2))
            for ridge in (1e-3, 1e-2, 1e-1)]
        self.assertLess(deviations[0], deviations[1])
        self.assertLess(deviations[1], deviations[2])


class TruncationTest(unittest.TestCase):
    def test_rank_deficient_gram_reports_its_retained_rank(self):
        values = torch.tensor([1.0, 0.5, 1e-9, 1e-12], dtype=torch.float64)
        basis = _orthonormal(4, 77)
        gram = (basis * values) @ basis.transpose(0, 1)
        inverse, retained = spectral_pseudo_inverse(gram, -1.0)
        self.assertEqual(retained, 2)
        self.assertTrue(torch.isfinite(inverse).all())

    def test_all_zero_gram_is_refused(self):
        with self.assertRaises(ValueError):
            spectral_pseudo_inverse(torch.zeros(3, 3, dtype=torch.float64))


class AttributionTest(unittest.TestCase):
    def test_attribution_walks_from_surrogate_to_corrected(self):
        edges, c_dir = _chain(seed=17)
        dim = edges[0].shape[0]
        maps = [_orthonormal(dim, 60 + level) @ torch.diag(
            torch.linspace(0.7, 1.4, dim, dtype=torch.float64)) for level in range(3)]
        moved_edges = [maps[i].transpose(0, 1) @ edges[i] @ maps[i + 1] for i in range(2)]
        moved_dir = maps[0].transpose(0, 1) @ c_dir @ maps[-1]
        correction = build_correction([m.transpose(0, 1) @ m for m in maps])
        steps = cumulative_interface_attribution(moved_edges, moved_dir, correction)
        self.assertEqual([s["interfaces_corrected"] for s in steps], [0, 1])
        surrogate = float(torch.linalg.matrix_norm(
            moved_dir - compose_edge_operators(moved_edges), ord="fro"))
        self.assertAlmostEqual(steps[0]["delta_frobenius"], surrogate, places=8)
        fully = float(torch.linalg.matrix_norm(
            corrected_endpoint(moved_dir, correction)
            - corrected_composition(moved_edges, correction), ord="fro"))
        self.assertAlmostEqual(steps[-1]["delta_frobenius"], fully, places=8)


class FiniteSampleTest(unittest.TestCase):
    def test_radius_shrinks_like_one_over_sqrt_n(self):
        generator = torch.Generator().manual_seed(5)
        radii = []
        for samples in (1000, 4000, 16000):
            left = torch.randn(samples, 8, generator=generator, dtype=torch.float64)
            right = torch.randn(samples, 8, generator=generator, dtype=torch.float64)
            radii.append(matrix_bernstein_radius(left, right).radius)
        self.assertLess(radii[1], radii[0])
        self.assertLess(radii[2], radii[1])
        # Bernstein is dominated by the sqrt term here, so a 16x sample
        # increase should buy better than a factor of two.
        self.assertLess(radii[2], radii[0] / 2)

    def test_path_radius_compounds(self):
        self.assertAlmostEqual(path_radius([0.1, 0.1]), 0.21, places=10)
        self.assertAlmostEqual(path_radius([0.0, 0.0]), 0.0, places=12)

    def test_tier2_never_exceeds_tier1(self):
        c_comp = torch.diag(torch.tensor([1.0, 0.7, 0.4], dtype=torch.float64))
        certificate = tiered_certificate(c_comp, delta_operator=0.05,
                                         endpoint_radius=0.02, edge_radii=[0.03, 0.04])
        self.assertTrue(bool((certificate.tier2 <= certificate.tier1 + 1e-12).all()))
        self.assertGreater(certificate.path_radius, 0.0)

    def test_clipping_is_counted_not_hidden(self):
        left = torch.ones(100, 3, dtype=torch.float64)
        right = torch.ones(100, 3, dtype=torch.float64)
        left[0] *= 50.0
        result = matrix_bernstein_radius(left, right, margin=1.0)
        self.assertEqual(result.clipped, 0)
        self.assertGreater(result.norm_bound, 0.0)


if __name__ == "__main__":
    unittest.main()
