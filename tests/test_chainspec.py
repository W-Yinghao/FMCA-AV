"""ChainSpec must enforce the object-correctness constraints itself."""

import unittest

import torch

from fmca_av.certificate.chainspec import ChainSpec, measure_chain, shuffled_null


def _spec(**kw):
    base = dict(name="t", interpretation="depth_sufficiency",
                state_names=["a", "b", "c"], deterministic_edges=True)
    base.update(kw)
    return ChainSpec(**base)


def _deterministic_chain(n=4000, dim=8, seed=0):
    """H0 -> H1 -> H2 with deterministic linear edges plus noise."""

    g = torch.Generator().manual_seed(seed)
    h0 = torch.randn(n, dim, generator=g, dtype=torch.float64)
    m1 = torch.randn(dim, dim, generator=g, dtype=torch.float64) / dim ** 0.5
    m2 = torch.randn(dim, dim, generator=g, dtype=torch.float64) / dim ** 0.5
    h1 = torch.tanh(h0 @ m1)
    h2 = torch.tanh(h1 @ m2)
    return [h0, h1, h2]


class SpecContractTest(unittest.TestCase):
    def test_undeclared_interpretation_is_refused(self):
        with self.assertRaises(ValueError):
            _spec(interpretation="physical_adapter")

    def test_two_state_chain_is_refused(self):
        with self.assertRaises(ValueError):
            _spec(state_names=["a", "b"])

    def test_state_count_must_match_the_data(self):
        states = _deterministic_chain()
        with self.assertRaises(ValueError):
            measure_chain(states[:2], states, _spec())


def _widening_chain(n=4000, dims=(6, 16, 40), seed=1):
    """The shape a real network has: every level a different width."""

    g = torch.Generator().manual_seed(seed)
    states = [torch.randn(n, dims[0], generator=g, dtype=torch.float64)]
    for previous, width in zip(dims, dims[1:]):
        m = torch.randn(previous, width, generator=g, dtype=torch.float64) / previous ** 0.5
        states.append(torch.tanh(states[-1] @ m))
    return states


class HeterogeneousWidthTest(unittest.TestCase):
    """Regression: levels of unequal width crashed the Gram correction.

    Every gate hierarchy so far used one width at every level, so the
    single shared identity in build_correction was vacuously right.  A
    pretrained depth chain is 64 -> 256 -> ... -> 2048 and broke it.
    """

    def test_unequal_widths_are_measured_not_refused(self):
        states = _widening_chain()
        half = states[0].shape[0] // 2
        out = measure_chain([s[:half] for s in states], [s[half:] for s in states], _spec())
        self.assertEqual(out["dimensions"], [6, 16, 40])
        self.assertEqual(out["gram"]["dimensions"], [6, 16, 40])
        self.assertTrue(out["projection"]["endpoint_top"] > 0.0)
        # The attribution walk switches interfaces on one at a time; with
        # unequal widths its identity placeholders must be per-level too.
        self.assertEqual([s["interfaces_corrected"] for s in out["interface_attribution"]],
                         [0, 1])

    def test_budget_matches_the_levels_and_reports_what_it_dropped(self):
        states = _widening_chain()
        half = states[0].shape[0] // 2
        spec = _spec(coordinate_budget=6)
        out = measure_chain([s[:half] for s in states], [s[half:] for s in states], spec)
        self.assertEqual(out["dimensions"], [6, 6, 6])
        self.assertEqual(out["native_dimensions"], [6, 16, 40])
        # Level 0 is untouched, the widened levels give up variance and say so.
        self.assertAlmostEqual(out["budget_retained_variance"][0], 1.0, places=12)
        for retained in out["budget_retained_variance"][1:]:
            self.assertTrue(0.0 < retained < 1.0)

    def test_budget_basis_is_fitted_on_calibration_only(self):
        """Evaluation data must not steer the projection it is measured in."""

        states = _widening_chain(seed=8)
        half = states[0].shape[0] // 2
        spec = _spec(coordinate_budget=8)
        calibration = [s[:half] for s in states]
        baseline = measure_chain(calibration, [s[half:] for s in states], spec)
        # Perturbing evaluation must move the readout; perturbing it must
        # NOT be able to re-fit the basis, so a scale on evaluation alone
        # cannot leave the answer invariant the way a re-fit would.
        scaled = [s[half:] * 3.0 for s in states]
        moved = measure_chain(calibration, scaled, spec)
        self.assertGreater(abs(moved["projection"]["endpoint_top"]
                               - baseline["projection"]["endpoint_top"]), 1e-6)


class ConditioningGuardTest(unittest.TestCase):
    """The variance floor must bound the Gram, or the correction explodes.

    Pooled activations have heavy-tailed spectra.  Without the floor the
    weakest retained direction reaches the correction as a near-zero Gram
    eigenvalue and G^{-1/2} amplifies it, which is how a composed operator
    ends up an order of magnitude above the endpoint it is supposed to
    reproduce.
    """

    def _skewed_chain(self, n=4000, dim=24, decay=1e-5, seed=3):
        g = torch.Generator().manual_seed(seed)
        spectrum = torch.logspace(0, torch.log10(torch.tensor(decay)).item(), dim,
                                  dtype=torch.float64)
        states = [torch.randn(n, dim, generator=g, dtype=torch.float64) * spectrum.sqrt()]
        for _ in range(2):
            m = torch.randn(dim, dim, generator=g, dtype=torch.float64) / dim ** 0.5
            states.append(torch.tanh(states[-1] @ m) * spectrum.sqrt())
        return states

    def test_floor_bounds_the_gram_and_the_composed_operator(self):
        states = self._skewed_chain()
        half = states[0].shape[0] // 2
        calibration = [s[:half] for s in states]
        evaluation = [s[half:] for s in states]

        loose = measure_chain(calibration, evaluation, _spec())
        floored = measure_chain(calibration, evaluation, _spec(variance_floor=1e-2))

        # The guarantee: lambda >= floor * lambda_max sends every Gram
        # eigenvalue above floor / (floor + ridge), so ||I - G|| stays well
        # under one.
        self.assertLess(max(floored["gram"]["metric_deviation"]), 0.15)
        # The mutation this pins: without the floor the Gram degenerates.
        self.assertGreater(max(loose["gram"]["metric_deviation"]), 0.5)
        # And the composed operator stays commensurate with the endpoint it
        # is supposed to reproduce, instead of drifting above it.
        def ratio(out):
            return out["projection"]["path_top"] / out["projection"]["endpoint_top"]
        self.assertLess(ratio(floored), ratio(loose))
        # Fewer coordinates survive, and the report says how many.
        self.assertLess(sum(floored["gram"]["retained_ranks"]),
                        sum(loose["gram"]["retained_ranks"]))

    def test_a_floor_that_retains_nothing_is_refused_not_silently_emptied(self):
        states = self._skewed_chain()
        half = states[0].shape[0] // 2
        with self.assertRaises(ValueError):
            measure_chain([s[:half] for s in states], [s[half:] for s in states],
                          _spec(variance_floor=0.9999))


class MeasurementTest(unittest.TestCase):
    def test_reports_every_required_component(self):
        states = _deterministic_chain()
        half = states[0].shape[0] // 2
        out = measure_chain([s[:half] for s in states], [s[half:] for s in states], _spec())
        for key in ("endpoint_top", "endpoint_mass", "endpoint_effective_rank",
                    "path_top", "path_mass", "delta_frobenius", "delta_operator"):
            self.assertIn(key, out["projection"])
            self.assertIn(key, out["surrogate"])
        self.assertEqual(out["calibration_samples"], half)
        self.assertEqual(out["evaluation_samples"], half)
        self.assertIn("retained_ranks", out["gram"])

    def test_shuffled_null_destroys_endpoint_dependence(self):
        states = _deterministic_chain(seed=2)
        half = states[0].shape[0] // 2
        spec = _spec()
        real = measure_chain([s[:half] for s in states], [s[half:] for s in states], spec)
        null = shuffled_null([s[:half] for s in states], [s[half:] for s in states], spec)
        self.assertGreater(real["projection"]["endpoint_top"],
                           3 * null["projection"]["endpoint_top"])

    def test_centering_is_not_optional(self):
        """A constant mode must not manufacture transport."""

        states = _deterministic_chain(seed=4)
        shifted = [s + 50.0 for s in states]
        half = states[0].shape[0] // 2
        spec = _spec()
        plain = measure_chain([s[:half] for s in states], [s[half:] for s in states], spec)
        offset = measure_chain([s[:half] for s in shifted], [s[half:] for s in shifted], spec)
        self.assertAlmostEqual(plain["projection"]["endpoint_top"],
                               offset["projection"]["endpoint_top"], places=6)


if __name__ == "__main__":
    unittest.main()
