"""Regression checks for the post-fix E6 aggregation boundary."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION
from scripts.render_e6_generalization_assets import has_current_source as generalization_source
from scripts.render_e6_robustness_assets import has_current_source as robustness_source
from scripts.render_e9_localization_assets import has_current_source as localization_source


class E6ResultVersioningTests(unittest.TestCase):
    def payload(self, checkpoint: Path) -> dict[str, str]:
        return {
            "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
            "source_checkpoint": str(checkpoint),
        }

    def test_requires_current_source_training_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "run" / "artifacts"
            checkpoint = artifacts / "checkpoints" / "last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            metadata = artifacts / "train_result.json"
            metadata.write_text(json.dumps({
                "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
            }), encoding="utf-8")
            value = self.payload(checkpoint)
            self.assertTrue(generalization_source(value))
            self.assertTrue(robustness_source(value))
            self.assertTrue(localization_source(value))

            metadata.write_text(json.dumps({}), encoding="utf-8")
            self.assertFalse(generalization_source(value))
            self.assertFalse(robustness_source(value))
            self.assertFalse(localization_source(value))

    def test_rejects_unversioned_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "run" / "artifacts"
            checkpoint = artifacts / "checkpoints" / "last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            (artifacts / "train_result.json").write_text(json.dumps({
                "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
            }), encoding="utf-8")
            value = {"source_checkpoint": str(checkpoint)}
            self.assertFalse(generalization_source(value))
            self.assertFalse(robustness_source(value))
            self.assertFalse(localization_source(value))

    def test_accepts_current_supervised_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "run" / "artifacts"
            checkpoint = artifacts / "checkpoints" / "last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            (artifacts / "supervised_result.json").write_text(json.dumps({
                "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
            }), encoding="utf-8")
            self.assertTrue(localization_source(self.payload(checkpoint)))


if __name__ == "__main__":
    unittest.main()
