import json
from pathlib import Path
import unittest

from scripts.launch_priority_checkpoint_probes import candidates


class PrioritySubmissionTests(unittest.TestCase):
    def test_candidate_gate_requires_paired_first_three_seeds(self) -> None:
        completed = []
        for views in (2, 8):
            for seed_index in range(1, 6):
                completed.append({
                    "action": {
                        "dataset": "cifar10", "method": "fmca_av", "kind": "train",
                        "views": views, "seed_index": seed_index, "seed": 20280000 + seed_index,
                        "target": 200, "key": f"cifar10:{views}:fmca_av:{seed_index}",
                    },
                    "run_id": f"run-{views}-{seed_index}", "state": "SUCCEEDED",
                })
        completed.extend([
            {"action": {"dataset": "cifar100", "method": "fmca_av", "kind": "train",
                        "views": 2, "seed_index": 1, "target": 200, "key": "wrong-dataset"},
             "run_id": "wrong-dataset", "state": "SUCCEEDED"},
            {"action": {"dataset": "cifar10", "method": "simclr", "kind": "train",
                        "views": 2, "seed_index": 1, "target": 200, "key": "wrong-method"},
             "run_id": "wrong-method", "state": "SUCCEEDED"},
        ])
        selected = candidates({"completed": completed})
        self.assertEqual(len(selected), 6)
        self.assertEqual(
            [(item["action"]["seed_index"], item["action"]["views"]) for item in selected],
            [(1, 2), (1, 8), (2, 2), (2, 8), (3, 2), (3, 8)],
        )

    def test_priority_policy_is_bounded_and_versioned(self) -> None:
        policy = json.loads(Path("configs/experiments/tpami_priority_20260809.json").read_text())
        allocation = policy["resource_allocation"]
        self.assertAlmostEqual(sum(allocation.values()), 1.0)
        self.assertEqual(policy["screening"]["paired_seed_indices"], [1, 2, 3])
        self.assertEqual(policy["screening"]["selection_gate"]["additional_seed_indices_for_selected"], [4, 5])
        self.assertIn("e7", policy["paused_expansions"])
        self.assertIn("e9", policy["paused_expansions"])


if __name__ == "__main__":
    unittest.main()
