import unittest

from scripts.render_priority_fmca_gate import choose


class PriorityFMCAGateTests(unittest.TestCase):
    def test_one_point_gate_and_compute_tie_break(self) -> None:
        rows = [
            {"configuration": "m2", "test_accuracy_mean": 0.90, "gpu_hours_mean": 2,
             "encoded_views_mean": 20},
            {"configuration": "m8", "test_accuracy_mean": 0.92, "gpu_hours_mean": 6,
             "encoded_views_mean": 80},
            {"configuration": "m4", "test_accuracy_mean": 0.915, "gpu_hours_mean": 3,
             "encoded_views_mean": 40},
        ]
        self.assertEqual(choose(rows), ["m8", "m4"])


if __name__ == "__main__":
    unittest.main()
