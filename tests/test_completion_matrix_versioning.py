"""Ensure the final completion gate cannot consume legacy result paths."""

from __future__ import annotations

import unittest

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION
from scripts.build_experiment_completion_matrix import build_requirements


class CompletionMatrixVersioningTests(unittest.TestCase):
    def test_all_result_paths_are_postfix_and_states_are_versioned(self) -> None:
        marker = f"results/postfix/{SCIENTIFIC_CORRECTNESS_VERSION}/"
        requirements = build_requirements()
        self.assertEqual(set(requirements), {f"E{index}" for index in range(11)})
        for files, states in requirements.values():
            self.assertTrue(files)
            self.assertTrue(all(path.startswith(marker) for path in files))
            for path in states:
                self.assertTrue(
                    SCIENTIFIC_CORRECTNESS_VERSION in path
                    or path.endswith("formal_ssl_postfix_state.json")
                )


if __name__ == "__main__":
    unittest.main()
