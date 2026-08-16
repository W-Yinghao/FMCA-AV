"""Frozen training objective: term semantics and gradient flow."""

import unittest

import torch

from fmca_av.certificate.counterexamples import (
    case_closed_chain,
    case_hallucinated_path,
    case_zero_operator,
)
from fmca_av.certificate.estimation import ChainFeatureBatch
from fmca_av.certificate.objective import certificate_training_loss


def _sample(case, parents: int, seed: int) -> ChainFeatureBatch:
    return case.sample(parents, children_per_edge=4, endpoint_descendants=4,
                       generator=torch.Generator().manual_seed(seed))


class ObjectiveSemanticsTests(unittest.TestCase):
    def test_closed_chain_has_small_closure_ratio(self) -> None:
        terms = certificate_training_loss(_sample(case_closed_chain(), 8000, seed=1))
        self.assertLess(float(terms.closure_ratio), 0.05)
        # Dimension-normalized score: ||C_dir||_F^2 / K = 0.735 / 3.
        self.assertGreater(float(terms.endpoint_score), 0.2)

    def test_hallucinated_path_is_punished_by_the_closure_term(self) -> None:
        terms = certificate_training_loss(_sample(case_hallucinated_path(), 8000, seed=2))
        # The endpoint dependence is zero while the ledger claims 1/2, so the
        # normalized closure ratio must explode relative to the closed chain.
        self.assertLess(float(terms.endpoint_score), 0.01)
        self.assertGreater(float(terms.closure_ratio), 5.0)

    def test_zero_operator_gets_no_reward_from_closure_alone(self) -> None:
        terms = certificate_training_loss(_sample(case_zero_operator(), 8000, seed=3))
        self.assertLess(float(terms.endpoint_score), 0.01)
        # Perfect closure with zero transmission: the only reward channel is
        # the endpoint term, which is (near) zero here by construction.
        self.assertGreater(float(terms.total), -0.05)

    def test_loss_terms_are_invariant_to_constant_feature_shifts(self) -> None:
        """Pins train-time centering: with the batch means subtracted, every
        operator term must be invariant to adding a constant vector to all
        features of a level; without centering the constant mode leaks into
        the operators and the whitening penalty."""

        batch = _sample(case_closed_chain(), 6000, seed=8)
        shifted = ChainFeatureBatch(
            chain=[states + 4.0 for states in batch.chain],
            children=[descendants + 4.0 for descendants in batch.children],
            endpoint_descendants=batch.endpoint_descendants + 4.0,
        )
        base = certificate_training_loss(batch, alpha=0.5)
        moved = certificate_training_loss(shifted, alpha=0.5)
        self.assertAlmostEqual(float(base.endpoint_score), float(moved.endpoint_score), places=6)
        self.assertAlmostEqual(float(base.closure_ratio), float(moved.closure_ratio), places=6)
        self.assertAlmostEqual(float(base.edge_score_sum), float(moved.edge_score_sum), places=6)
        self.assertAlmostEqual(
            float(base.whitening_penalty), float(moved.whitening_penalty), places=6
        )

    def test_whitening_penalty_reacts_to_covariance_inflation(self) -> None:
        batch = _sample(case_closed_chain(), 4000, seed=4)
        inflated = ChainFeatureBatch(
            chain=[2.0 * states for states in batch.chain],
            children=[2.0 * descendants for descendants in batch.children],
            endpoint_descendants=2.0 * batch.endpoint_descendants,
        )
        base = certificate_training_loss(batch)
        scaled = certificate_training_loss(inflated)
        self.assertGreater(float(scaled.whitening_penalty), float(base.whitening_penalty) + 1.0)

    def test_alpha_reproduces_the_additive_family_sign(self) -> None:
        batch = _sample(case_closed_chain(), 4000, seed=5)
        without = certificate_training_loss(batch, alpha=0.0, beta=0.0, gamma=0.0)
        with_alpha = certificate_training_loss(batch, alpha=1.0, beta=0.0, gamma=0.0)
        # Positive alpha must REWARD per-edge dependence (maximizing sign).
        self.assertLess(float(with_alpha.total), float(without.total))


class GradientFlowTests(unittest.TestCase):
    def test_gradients_flow_through_every_term(self) -> None:
        batch = _sample(case_closed_chain(), 512, seed=6)
        chain = [states.clone().requires_grad_(True) for states in batch.chain]
        children = [descendants.clone().requires_grad_(True) for descendants in batch.children]
        endpoint = batch.endpoint_descendants.clone().requires_grad_(True)
        differentiable = ChainFeatureBatch(chain=chain, children=children, endpoint_descendants=endpoint)
        terms = certificate_training_loss(differentiable, alpha=0.1, beta=1.0, gamma=1.0)
        terms.total.backward()
        for tensor in [*chain, *children, endpoint]:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().max()), 0.0)

    def test_closure_stop_grad_detaches_the_direct_branch_in_the_ratio(self) -> None:
        batch = _sample(case_closed_chain(), 512, seed=7)
        children = [descendants.clone().requires_grad_(True) for descendants in batch.children]
        endpoint = batch.endpoint_descendants.clone().requires_grad_(True)
        differentiable = ChainFeatureBatch(
            chain=batch.chain, children=children, endpoint_descendants=endpoint
        )
        terms = certificate_training_loss(
            differentiable, alpha=0.0, beta=1.0, gamma=0.0, closure_stop_grad=True
        )
        terms.closure_ratio.backward(retain_graph=True)
        # With the endpoint branch detached inside the ratio, only the chain
        # children (through C_comp) receive closure gradient.
        self.assertIsNone(endpoint.grad)
        for descendants in children:
            self.assertIsNotNone(descendants.grad)
            self.assertGreater(float(descendants.grad.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
