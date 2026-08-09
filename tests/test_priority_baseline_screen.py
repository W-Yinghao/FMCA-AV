import unittest

from scripts.launch_priority_baseline_screen import parse_initial, select_methods


class PriorityBaselineScreenTests(unittest.TestCase):
    def test_selection_uses_accuracy_gate_then_gpu_hours(self) -> None:
        rows = []
        for method, accuracy, gpu_hours in (
            ("simclr", 0.80, 2.0), ("vicreg", 0.805, 3.0),
            ("dino", 0.805, 4.0), ("byol", 0.78, 1.0),
        ):
            rows.extend({"method": method, "test_accuracy": accuracy,
                         "gpu_hours": gpu_hours} for _ in range(3))
        self.assertEqual(select_methods(rows), ["vicreg", "dino"])

    def test_initial_run_parser(self) -> None:
        self.assertEqual(parse_initial("simclr:1=run-a,vicreg:1=run-b"),
                         {"simclr:1": "run-a", "vicreg:1": "run-b"})


if __name__ == "__main__":
    unittest.main()
