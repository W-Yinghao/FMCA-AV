import glob
import json
import unittest
from pathlib import Path

from fmca_av.certificate.hierarchy_module import HierarchyCertificateModule


def build(path, **loss_overrides):
    config = json.loads(Path(path).read_text())
    if loss_overrides:
        config["loss"] = {**config.get("loss", {}), **loss_overrides}
    return HierarchyCertificateModule(config)


class EstimatorSwitchTests(unittest.TestCase):
    def test_every_config_on_disk_resolves_as_it_did_before(self) -> None:
        """The prefix rule, reproduced exactly for configs with no key.

        This is the guarantee that adding the switch re-labels nothing in
        the existing corpus: 26 MAJOR units and 286 ridge ones keep the
        estimator they were actually trained under.
        """

        paths = sorted(glob.glob("configs/gate*/*.json"))
        self.assertGreater(len(paths), 50)
        for path in paths:
            config = json.loads(Path(path).read_text())
            self.assertNotIn(
                "estimator", config.get("loss", {}),
                msg=f"{path} carries an explicit estimator; this test assumes none do")
            variant = config.get("variant", "product_endpoint")
            expected = "truncated" if variant.startswith("paper_") else "ridge"
            with self.subTest(path=path):
                self.assertEqual(build(path).estimator, expected)

    def test_switch_decouples_estimator_from_recipe(self) -> None:
        v8 = "configs/gate_v8/gate1_cifar10_v7_product_endpoint.json"
        paper = "configs/gate_paper/gate1_cifar10_paper_composition.json"
        # The two cells that did not exist before.
        self.assertEqual(build(v8, estimator="truncated").estimator, "truncated")
        self.assertEqual(build(paper, estimator="ridge").estimator, "ridge")
        # The variant is untouched by the estimator choice.
        self.assertEqual(build(v8, estimator="truncated").variant, "product_endpoint")

    def test_bad_estimator_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            build("configs/gate_paper/gate1_cifar10_paper_composition.json",
                  estimator="whitened")

    def test_method_assertions_still_bind_paper_arms(self) -> None:
        """Changing the estimator must not loosen the paper's method spec."""
        with self.assertRaises(ValueError):
            build("configs/gate_paper/gate1_cifar10_paper_composition.json",
                  estimator="ridge", ema_target_momentum=0.99)


if __name__ == "__main__":
    unittest.main()
