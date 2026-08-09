import unittest

from scripts.e4_priority_designs import DESIGNS, encoded_forwards, override


class E4PriorityDesignTests(unittest.TestCase):
    def test_all_designs_match_eight_encoded_forwards(self) -> None:
        self.assertEqual(set(DESIGNS), {"raw_parent", "mean", "deepsets", "concat"})
        self.assertTrue(all(encoded_forwards(design) == 8 for design in DESIGNS))

    def test_raw_parent_uses_seven_conditionals_plus_parent(self) -> None:
        value = override("raw_parent", 2)
        self.assertEqual(value["data"]["num_views"], 7)
        self.assertTrue(value["data"]["include_raw_parent"])
        self.assertEqual(value["model"]["parent_aggregation"], "raw")

    def test_unknown_design_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            override("unknown", 1)


if __name__ == "__main__":
    unittest.main()
