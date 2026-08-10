import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness import cli, state


class HarnessBudgetConfigTest(unittest.TestCase):
    def setUp(self):
        self.base = json.loads(state.CONFIG_PATH.read_text(encoding="utf-8"))

    def load(self, **updates):
        config = copy.deepcopy(self.base)
        config.update(updates)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with mock.patch.object(state, "CONFIG_PATH", path):
                return state.load_config()

    def test_six_and_eight_gpu_aggregate_budgets_are_valid(self):
        self.assertEqual(self.load(max_gpus=6)["max_gpus"], 6)
        self.assertEqual(self.load(max_gpus=8)["max_gpus"], 8)

    def test_budget_above_temporary_ceiling_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "either 6 or 8"):
            self.load(max_gpus=9)

    def test_per_job_limit_remains_two(self):
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            self.load(max_gpus_per_job=3)

    def test_allowed_ids_cannot_exceed_aggregate_budget(self):
        self.assertEqual(
            len(self.load(max_gpus=8, allowed_gpu_ids=list(range(8)))["allowed_gpu_ids"]),
            8,
        )
        with self.assertRaisesRegex(ValueError, "configured GPU budget"):
            self.load(max_gpus=6, allowed_gpu_ids=list(range(7)))

    def test_named_extra_pool_is_isolated_from_default_budget(self):
        config = self.load()
        jobs = {"jobs": {
            "regular": {"state": "RUNNING", "requested_gpus": 6},
            "toy": {"state": "RUNNING", "requested_gpus": 1,
                    "gpu_budget_pool": "nonlinear_neural"},
        }}
        _, regular_used = cli.allocation_for(config, "slurm", 0, jobs, "regular-job")
        _, toy_used = cli.allocation_for(
            config, "slurm", 1, jobs, "e1-neural-nonlinear-two-moons",
        )
        self.assertEqual(regular_used, 6)
        self.assertEqual(toy_used, 1)
        with self.assertRaisesRegex(RuntimeError, "nonlinear_neural"):
            cli.allocation_for(config, "slurm", 2, jobs, "e1-neural-nonlinear-gmm")


if __name__ == "__main__":
    unittest.main()
