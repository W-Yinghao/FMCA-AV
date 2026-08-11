import unittest

from scripts import run_external_multiview_baselines as controller


class ExternalBaselineControllerTests(unittest.TestCase):
    def test_scoped_matrix_is_complete_and_cifar10_only(self) -> None:
        actions = controller.actions()
        self.assertEqual(controller.POLL_SECONDS, 300)
        self.assertEqual(len(actions), 82)
        methods = {str(action["method"]) for action in actions if "method" in action}
        self.assertEqual(methods, {"fastssl_barlow_twins", "fastssl_vicreg", "frossl"})
        trains = [action for action in actions if action["kind"] == "train"]
        self.assertEqual(len(trains), 18)
        self.assertEqual({int(action["views"]) for action in trains}, {2, 8})
        self.assertEqual({int(action["seed"]) for action in trains}, set(controller.SEEDS))
        self.assertTrue(all("cifar10" in str(action["config"]) for action in trains))

    def test_frossl_view_dependent_gamma_and_fastssl_no_fake_override(self) -> None:
        self.assertEqual(controller.method_override("frossl", 2, 1)["objective"]["invariance_weight"], 1.4)
        self.assertEqual(controller.method_override("frossl", 8, 1)["objective"]["invariance_weight"], 2.0)
        self.assertNotIn("objective", controller.method_override("fastssl_barlow_twins", 8, 1))


if __name__ == "__main__":
    unittest.main()
