import copy
import json
import unittest
from pathlib import Path

import torch

from fmca_av.baselines import BaselineSSL

REPO = Path(__file__).resolve().parent.parent
BASE = json.loads((REPO / "configs/ssl_matched/cifar10_byol.json").read_text())


def build(method, views, seed=0):
    config = copy.deepcopy(BASE)
    config["experiment"]["method"] = method
    config["data"]["num_views"] = views
    config["model"]["head_normalization"] = "batch"
    torch.manual_seed(seed)
    return BaselineSSL(config).train()


def batch(views, n=8, seed=1):
    torch.manual_seed(seed)
    return (torch.randn(n, views, 3, 32, 32), torch.zeros(n, dtype=torch.long), torch.zeros(n))


class SiamBaselineTests(unittest.TestCase):
    def test_at_two_views_fastsiam_equals_simsiam(self) -> None:
        """The mean of the one other view IS that view.

        This identity is why the difference stayed invisible: the repo ran
        fastsiam only at two views, where the pair loop happened to be right.
        """
        fast, siam = build("fastsiam", 2), build("simsiam", 2)
        siam.load_state_dict(fast.state_dict())
        with torch.no_grad():
            a = fast._shared_step(batch(2), "train")
            b = siam._shared_step(batch(2), "train")
        self.assertAlmostEqual(float(a), float(b), places=5)

    def test_at_four_views_fastsiam_differs_from_paired_simsiam(self) -> None:
        """Beyond two views the two are different methods, not a refactor."""
        fast, siam = build("fastsiam", 4), build("simsiam", 4)
        siam.load_state_dict(fast.state_dict())
        with torch.no_grad():
            a = fast._shared_step(batch(4), "train")
            b = siam._shared_step(batch(4), "train")
        self.assertNotAlmostEqual(float(a), float(b), places=4)

    def test_fastsiam_target_is_the_mean_of_the_others(self) -> None:
        model = build("fastsiam", 4)
        with torch.no_grad():
            views = batch(4)[0]
            encoded = model.backbone(views.flatten(0, 1))
            projections = model.projector(encoded).reshape(views.shape[0], 4, -1)
            expected = torch.stack([
                model._negative_cosine(
                    model.predictor(projections[:, i]),
                    projections[:, [j for j in range(4) if j != i]].mean(dim=1))
                for i in range(4)]).mean()
            got = model._shared_step(batch(4), "train")
        self.assertAlmostEqual(float(got), float(expected), places=5)

    def test_both_get_a_predictor(self) -> None:
        for method in ("simsiam", "fastsiam"):
            self.assertIsNotNone(build(method, 2).predictor, msg=method)


if __name__ == "__main__":
    unittest.main()
