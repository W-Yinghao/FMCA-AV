import unittest

from scripts.continue_priority_fmca_v8 import action


class PriorityContinuationTests(unittest.TestCase):
    def test_selected_chain_actions_are_bounded(self) -> None:
        train = action(2, "train", 600)
        self.assertEqual(train["dataset"], "cifar10")
        self.assertEqual(train["method"], "fmca_av")
        self.assertEqual(train["views"], 8)
        self.assertEqual(train["seed_index"], 2)
        self.assertEqual(train["target"], 600)
        self.assertNotIn("target", action(2, "probe"))

    def test_only_three_paired_seeds_exist(self) -> None:
        with self.assertRaises(IndexError):
            action(4, "train", 600)


if __name__ == "__main__":
    unittest.main()
