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
