"""Numerics-boundary checks for post-fix E10 inputs."""

from __future__ import annotations

import unittest

from fmca_av.operators import MOMENT_ACCUMULATION_POLICY
from scripts.render_complexity_assets import require_current_moment_policy


class ComplexityVersioningTests(unittest.TestCase):
    def test_renderer_requires_float32_moment_policy(self) -> None:
        require_current_moment_policy({"moment_accumulation_policy": MOMENT_ACCUMULATION_POLICY})
        with self.assertRaises(ValueError):
            require_current_moment_policy({})
        with self.assertRaises(ValueError):
            require_current_moment_policy({"moment_accumulation_policy": "float16"})


if __name__ == "__main__":
    unittest.main()
