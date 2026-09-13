import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import sweep_pending as sp
from fmca_av.profiling import MilestoneCheckpoint


class MilestoneWiringTests(unittest.TestCase):
    def test_runner_imports_and_uses_the_milestone_callback(self) -> None:
        source = (REPO / "scripts" / "run_gate1_unit.py").read_text()
        self.assertIn("from fmca_av.profiling import MilestoneCheckpoint", source)
        self.assertIn("checkpoint_milestones", source)
        self.assertIn("MilestoneCheckpoint(unit_dir", source)

    def test_milestones_are_positive_epochs(self) -> None:
        with self.assertRaises(ValueError):
            MilestoneCheckpoint(Path("/tmp"), [0])
        MilestoneCheckpoint(Path("/tmp"), [20, 200, 800])


class ProgressAwareRetryTests(unittest.TestCase):
    """A walltime kill on a resumable run must not burn the retry cap.

    Every GPU partition here caps at 24h and an 800-epoch unit needs more,
    so being killed and resumed is the design, not a failure.
    """

    def setUp(self) -> None:
        self.attempts = sp.ATTEMPTS
        self.tmp = Path(REPO, "runs", ".attempts_test.tsv")
        sp.ATTEMPTS = self.tmp

    def tearDown(self) -> None:
        sp.ATTEMPTS = self.attempts
        self.tmp.unlink(missing_ok=True)

    def test_attempts_that_made_progress_do_not_count(self) -> None:
        key = "train:results/gate1/x:v:1"
        # two attempts, the second started from a later epoch than the first
        self.tmp.write_text(f"{key}\t100@-1\n{key}\t101@210\n")
        sp._progress_epoch = lambda k: 430          # progressed again since
        self.assertNotIn(key, sp.exhausted())

    def test_repeated_attempts_from_the_same_epoch_still_exhaust(self) -> None:
        key = "train:results/gate1/x:v:1"
        self.tmp.write_text(f"{key}\t100@-1\n{key}\t101@-1\n")
        sp._progress_epoch = lambda k: -1           # never got anywhere
        self.assertIn(key, sp.exhausted())

    def test_legacy_rows_without_progress_markers_still_count(self) -> None:
        key = "cert:results/gate1/x:v:1"
        self.tmp.write_text(f"{key}\t100\n{key}\t101\n")
        self.assertIn(key, sp.exhausted())


if __name__ == "__main__":
    unittest.main()
