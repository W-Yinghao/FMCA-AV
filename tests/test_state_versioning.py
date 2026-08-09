"""Regression checks for post-fix orchestration-state boundaries."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION
from scripts.formal_imagenet_state_machine import load_state as load_imagenet
from scripts.formal_imagenet_low_label_state_machine import load as load_imagenet_low_label
from scripts.formal_localization_state_machine import load as load_localization
from scripts.formal_transfer_state_machine import load as load_transfer


class StateVersioningTests(unittest.TestCase):
    def test_new_states_carry_current_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            states = (
                load_imagenet(root / "imagenet.json", "dependency"),
                load_transfer(root / "transfer.json", "pretrain.json"),
                load_localization(root / "localization.json", "pretrain.json"),
                load_imagenet_low_label(root / "lowlabel.json", "pretrain.json"),
            )
            for state in states:
                self.assertEqual(
                    state["scientific_correctness_version"],
                    SCIENTIFIC_CORRECTNESS_VERSION,
                )

    def test_legacy_states_are_rejected(self) -> None:
        loaders = (load_imagenet, load_transfer, load_localization, load_imagenet_low_label)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, loader in enumerate(loaders):
                path = root / f"legacy-{index}.json"
                path.write_text(json.dumps({"state": "RUNNING"}), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    loader(path, "dependency.json")


if __name__ == "__main__":
    unittest.main()
