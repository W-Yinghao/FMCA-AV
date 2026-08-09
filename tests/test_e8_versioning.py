"""Post-fix isolation checks for E8 Markov assets."""

from __future__ import annotations

from pathlib import Path
import unittest

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION
from scripts.launch_postfix_e8 import load
from scripts.render_e8_markov_assets import DEFAULT_OUTPUT, require_current_payload


class E8VersioningTests(unittest.TestCase):
    def test_renderer_rejects_legacy_source(self) -> None:
        with self.assertRaises(RuntimeError):
            require_current_payload({}, Path("legacy.json"))
        require_current_payload(
            {"scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION},
            Path("current.json"),
        )

    def test_default_output_is_postfix(self) -> None:
        self.assertEqual(
            DEFAULT_OUTPUT,
            Path("results/postfix") / SCIENTIFIC_CORRECTNESS_VERSION / "e8",
        )

    def test_new_state_is_versioned(self) -> None:
        state = load(Path("/definitely/missing/e8-state.json"))
        self.assertEqual(state["scientific_correctness_version"], SCIENTIFIC_CORRECTNESS_VERSION)


if __name__ == "__main__":
    unittest.main()
