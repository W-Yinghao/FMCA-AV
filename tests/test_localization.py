import unittest

import torch

from fmca_av.operators import dependence_contribution_maps


class DependenceContributionMapTests(unittest.TestCase):
    def test_map_changes_when_parent_f_changes_with_fixed_local_g(self) -> None:
        local = torch.tensor([[1.0, 0.5], [0.25, 2.0]], dtype=torch.float64)
        singular_values = torch.tensor([0.8, 0.3], dtype=torch.float64)
        first = dependence_contribution_maps(
            torch.tensor([1.0, 0.0], dtype=torch.float64), local, singular_values,
        )["signed_dependence"]
        second = dependence_contribution_maps(
            torch.tensor([0.0, 1.0], dtype=torch.float64), local, singular_values,
        )["signed_dependence"]
        self.assertFalse(torch.allclose(first, second))

    def test_zero_singular_values_give_zero_dependence_map(self) -> None:
        parent = torch.tensor([2.0, -3.0], dtype=torch.float64)
        local = torch.tensor([[4.0, 5.0], [-1.0, 7.0]], dtype=torch.float64)
        result = dependence_contribution_maps(
            parent, local, torch.zeros(2, dtype=torch.float64),
        )
        self.assertTrue(torch.equal(result["signed_dependence"], torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(torch.equal(result["absolute_dependence"], torch.zeros(2, dtype=torch.float64)))

    def test_toy_maximum_contribution_position_is_recovered(self) -> None:
        parent = torch.tensor([1.0, 2.0], dtype=torch.float64)
        local = torch.tensor(
            [[0.1, 0.0], [0.0, 0.2], [3.0, 4.0], [-0.5, 0.0]],
            dtype=torch.float64,
        )
        singular_values = torch.tensor([0.5, 0.25], dtype=torch.float64)
        absolute = dependence_contribution_maps(parent, local, singular_values)["absolute_dependence"]
        self.assertEqual(int(torch.argmax(absolute)), 2)


if __name__ == "__main__":
    unittest.main()
