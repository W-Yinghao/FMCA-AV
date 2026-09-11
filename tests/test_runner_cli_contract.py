"""Every encoding runner must offer --allow-cpu if it consults it.

The guard was added to five runners by one templated edit that keyed on a
line only four of them had.  The fifth got the USE without the FLAG, so
all eleven of its jobs died instantly on an AttributeError.  A templated
edit across files needs a check that each file got both halves.
"""

import ast
import pathlib
import unittest

RUNNERS = [
    "run_validity_gate.py", "run_layerwise_profile.py", "run_multiview_stability.py",
    "run_paper_certificate.py", "run_baseline_layerwise.py",
]


class RunnerCliContract(unittest.TestCase):
    def test_allow_cpu_is_declared_wherever_it_is_read(self):
        root = pathlib.Path(__file__).resolve().parent.parent / "scripts"
        for name in RUNNERS:
            source = (root / name).read_text()
            reads = "arguments.allow_cpu" in source
            declares = '"--allow-cpu"' in source
            self.assertEqual(reads, declares,
                             msg=f"{name}: reads={reads} declares={declares}")

    def test_every_runner_still_parses(self):
        root = pathlib.Path(__file__).resolve().parent.parent / "scripts"
        for name in RUNNERS:
            ast.parse((root / name).read_text())


if __name__ == "__main__":
    unittest.main()
