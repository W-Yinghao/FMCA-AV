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
from scripts.launch_postfix_factor_suite import load_state as load_factor, plan as factor_plan
from scripts.launch_e3_imagenet100_recheck import load_state as load_imagenet100_e3


class StateVersioningTests(unittest.TestCase):
    def test_factor_plan_has_frozen_coverage(self) -> None:
        records = factor_plan()
        self.assertEqual(len(records), 54)
        self.assertEqual(sum(record["channel"] == "default" for record in records), 18)
        self.assertEqual(len({record["key"] for record in records}), 54)

    def test_new_states_carry_current_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            states = (
                load_imagenet(root / "imagenet.json", "dependency"),
                load_transfer(root / "transfer.json", "pretrain.json"),
                load_localization(root / "localization.json", "pretrain.json"),
                load_imagenet_low_label(root / "lowlabel.json", "pretrain.json"),
                load_factor(root / "factor.json"),
                load_imagenet100_e3(root / "imagenet100-e3.json"),
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

            factor_path = root / "legacy-factor.json"
            factor_path.write_text(json.dumps({"state": "RUNNING"}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_factor(factor_path)

            e3_path = root / "legacy-imagenet100-e3.json"
            e3_path.write_text(json.dumps({"state": "RUNNING"}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_imagenet100_e3(e3_path)


if __name__ == "__main__":
    unittest.main()
