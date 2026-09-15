import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import sweep_pending
from run_gate1_unit import VARIANT_TAGS
from fmca_av.certificate.hierarchy_module import HierarchyCertificateModule


class SweepRuleTests(unittest.TestCase):
    def test_sweep_estimator_rule_matches_the_module(self) -> None:
        """The sweep re-implements the rule to avoid importing torch."""
        pairs = {(cfg, variant)
                 for cfg, _root, variant, _p, _s in sweep_pending.WAVE}
        pairs |= {(cfg, variant)
                  for cfg, _root, variant, _s, _g in sweep_pending.GATED_ELSEWHERE}
        pairs |= {(sweep_pending.X2X2_CONFIG[v], v) for v in sweep_pending.X2X2_CONFIG}
        self.assertGreater(len(pairs), 4)
        for config_dir, variant in sorted(pairs):
            path = REPO / config_dir / f"gate1_cifar10_{VARIANT_TAGS[variant]}.json"
            module = HierarchyCertificateModule(json.loads(path.read_text()))
            with self.subTest(config=config_dir, variant=variant):
                self.assertEqual(
                    sweep_pending.resolved_estimator(config_dir, variant),
                    module.estimator)

    def test_every_swept_root_has_a_certificate_tag(self) -> None:
        for root in sweep_pending.DERIVED_ROOTS:
            self.assertIn(root, sweep_pending.ROOT_TAG, msg=f"{root} has no tag")

    def test_no_two_real_units_map_to_one_certificate_file(self) -> None:
        """The 2x2 trains paper_composition too.

        An untagged name would have overwritten the MAJOR certificate for
        that variant and seed with one measured on a ridge-trained arm.
        paper_probe and paper_v1 deliberately SHARE the empty tag -- they
        partition the seeds of one logical CIFAR-10 gate, which is why the
        12 files on disk use the bare name -- so the invariant is over the
        units that actually exist, not over the cartesian product.
        """
        seen = {}
        for root, tag in sweep_pending.ROOT_TAG.items():
            units = REPO / root / "units"
            if not units.is_dir():
                continue
            for unit in sorted(units.glob("*__seed*")):
                variant, seed = unit.name.split("__seed")
                # Every variant, not just paper_ ones: the certificate rule
                # gates on the estimator now, so product_endpoint under
                # truncation writes a certificate too.
                name = f"{variant}{tag}_seed{seed}.json"
                self.assertNotIn(
                    name, seen,
                    msg=f"{name} claimed by both {seen.get(name)} and {root}")
                seen[name] = root
        self.assertGreater(len(seen), 20)

    def test_legacy_major_roots_keep_their_existing_filenames(self) -> None:
        existing = sorted(p.name for p in (REPO / "results" / "paper_certificate").glob("*.json"))
        self.assertIn("paper_composition_seed1.json", existing)
        self.assertEqual(sweep_pending.ROOT_TAG["results/gate1/gate1_20260910_paper_v1"], "")
        self.assertEqual(sweep_pending.ROOT_TAG["results/gate1/gate1_20260910_paper_probe"], "")


if __name__ == "__main__":
    unittest.main()
