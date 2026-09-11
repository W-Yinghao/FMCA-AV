import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_gate1_unit import DivergenceGuard


# The bounds the entire K=128 corpus was run under.  Pinned literally so a
# later edit to the guard cannot silently re-bound arms already on disk.
CORPUS_K128_BOUNDS = {
    "train/endpoint_score": 2.5,
    "train/leaf_score": 2.5,
    "train/edge_score_sum": 5.0,
    "train/cross_score_sum": 7.5,
    "train/product_score": 2.5,
    "train/flat_trace_score": 200.0,
    "train/dir_trace": 200.0,
    "train/leaf_trace": 200.0,
    "train/edge_trace_sum": 400.0,
    "train/cross_trace_sum": 600.0,
    "train/loss": 700.0,
}


class _Trainer:
    def __init__(self, **metrics):
        self.callback_metrics = {k: torch.tensor(v) for k, v in metrics.items()}


class DivergenceGuardTests(unittest.TestCase):
    def test_k128_bounds_are_unchanged_from_the_corpus(self) -> None:
        self.assertEqual(DivergenceGuard(128).bounds, CORPUS_K128_BOUNDS)

    def test_default_is_k128(self) -> None:
        self.assertEqual(DivergenceGuard().bounds, DivergenceGuard(128).bounds)

    def test_trace_bounds_scale_with_k_and_score_bounds_do_not(self) -> None:
        bounds = DivergenceGuard(256).bounds
        self.assertEqual(bounds["train/edge_trace_sum"], 800.0)
        self.assertEqual(bounds["train/dir_trace"], 400.0)
        self.assertEqual(bounds["train/endpoint_score"], 2.5)

    def test_healthy_k256_plateau_does_not_trip(self) -> None:
        # The composition arm plateaus at 211.8 with K=128; the same
        # behaviour at K=256 is ~423, which the old hard-coded 400 aborted.
        guard = DivergenceGuard(256)
        guard.on_train_epoch_end(_Trainer(**{"train/edge_trace_sum": 423.0}), None)

    def test_runaway_still_trips(self) -> None:
        guard = DivergenceGuard(256)
        with self.assertRaises(RuntimeError):
            guard.on_train_epoch_end(_Trainer(**{"train/edge_trace_sum": 900.0}), None)

    def test_non_finite_trips(self) -> None:
        guard = DivergenceGuard(128)
        with self.assertRaises(RuntimeError):
            guard.on_train_epoch_end(_Trainer(**{"train/loss": float("nan")}), None)


if __name__ == "__main__":
    unittest.main()
