"""Regression checks for post-fix orchestration-state boundaries."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fmca_av.operators import SCIENTIFIC_CORRECTNESS_VERSION
from scripts.formal_imagenet_state_machine import actions as imagenet_actions, load_state as load_imagenet
from scripts.formal_ssl_state_machine import formal_gpu_capacity
from scripts.launch_cifar10_tsd_sweep import BATCH_LIMIT as TSD_BATCH_LIMIT, LEVELS as TSD_LEVELS, SEEDS as TSD_SEEDS
from scripts.formal_imagenet_low_label_state_machine import load as load_imagenet_low_label
from scripts.formal_localization_state_machine import load as load_localization
from scripts.formal_transfer_state_machine import load as load_transfer
from scripts.launch_postfix_factor_suite import load_state as load_factor, plan as factor_plan
from scripts.launch_postfix_e10 import REQUIRED_DDP_GPUS, ddp_command
from scripts.launch_e3_imagenet100_recheck import load_state as load_imagenet100_e3
from scripts.launch_postfix_downstream import EXTERNAL_WATCHERS, load as load_downstream


class StateVersioningTests(unittest.TestCase):
    def test_cifar10_tsd_plan_is_complete_but_fairly_batched(self) -> None:
        self.assertEqual(TSD_BATCH_LIMIT, 2)
        self.assertEqual(sum(len(levels) for levels in TSD_LEVELS.values()) * len(TSD_SEEDS), 210)

    def test_e10_owns_only_one_and_two_gpu_scaling_points(self) -> None:
        self.assertEqual(REQUIRED_DDP_GPUS, (1, 2))
        self.assertNotIn("torchrun", " ".join(ddp_command(1)))
        self.assertIn("--nproc_per_node=2", ddp_command(2))
        self.assertFalse(any(action.get("kind") == "ddp" for action in imagenet_actions()))

    def test_imagenet_state_migrates_completed_ddp_prefix_to_e10(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "imagenet.json"
            path.write_text(json.dumps({
                "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
                "state": "RUNNING",
                "action_index": 1,
                "current_run": "",
                "current_action": None,
                "completed": [{
                    "action": {"kind": "ddp", "gpus": 1},
                    "run_id": "postfix-ddp1",
                    "state": "SUCCEEDED",
                }],
            }), encoding="utf-8")
            migrated = load_imagenet(path, "dependency")
            self.assertEqual(migrated["action_index"], 0)
            self.assertEqual(migrated["completed"], [])
            self.assertEqual(migrated["migrated_e10_ddp_runs"], ["postfix-ddp1"])

    def test_imagenet_dependency_uses_full_formal_state_not_watcher_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependency = root / "formal.json"
            dependency.write_text(json.dumps({
                "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
                "state": "RUNNING",
            }), encoding="utf-8")
            state_path = root / "imagenet.json"
            state_path.write_text(json.dumps({
                "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
                "state": "RUNNING",
                "dependency_complete": True,
                "action_index": 0,
                "current_run": "",
                "current_action": None,
                "completed": [],
            }), encoding="utf-8")
            self.assertFalse(load_imagenet(state_path, str(dependency))["dependency_complete"])

    def test_formal_ssl_capacity_reserves_global_slots(self) -> None:
        self.assertEqual(formal_gpu_capacity({"max_gpus": 6, "formal_ssl_max_gpus": 4}), 4)
        self.assertEqual(formal_gpu_capacity({"max_gpus": 3, "formal_ssl_max_gpus": 4}), 3)
        with self.assertRaises(ValueError):
            formal_gpu_capacity({"max_gpus": 6, "formal_ssl_max_gpus": 0})

    def test_factor_plan_has_frozen_coverage(self) -> None:
        records = factor_plan()
        self.assertEqual(len(records), 54)
        self.assertEqual(sum(record["channel"] == "default" for record in records), 18)
        self.assertEqual(len({record["key"] for record in records}), 54)

    def test_new_states_carry_current_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            states = (
                load_imagenet(root / "imagenet.json", "dependency"),
                load_transfer(root / "transfer.json", "pretrain.json"),
                load_localization(root / "localization.json", "pretrain.json"),
                load_imagenet_low_label(root / "lowlabel.json", "pretrain.json"),
                load_factor(root / "factor.json"),
                load_imagenet100_e3(root / "imagenet100-e3.json"),
                load_downstream(root / "downstream.json"),
            )
            for state in states:
                self.assertEqual(
                    state["scientific_correctness_version"],
                    SCIENTIFIC_CORRECTNESS_VERSION,
                )

    def test_legacy_states_are_rejected(self) -> None:
        loaders = (load_imagenet, load_transfer, load_localization, load_imagenet_low_label)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, loader in enumerate(loaders):
                path = root / f"legacy-{index}.json"
                path.write_text(json.dumps({"state": "RUNNING"}), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    loader(path, "dependency.json")

            factor_path = root / "legacy-factor.json"
            factor_path.write_text(json.dumps({"state": "RUNNING"}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_factor(factor_path)

            e3_path = root / "legacy-imagenet100-e3.json"
            e3_path.write_text(json.dumps({"state": "RUNNING"}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_imagenet100_e3(e3_path)

            downstream_path = root / "legacy-downstream.json"
            downstream_path.write_text(json.dumps({"state": "RUNNING"}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_downstream(downstream_path)

    def test_downstream_refreshes_operational_watcher_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "downstream.json"
            path.write_text(json.dumps({
                "scientific_correctness_version": SCIENTIFIC_CORRECTNESS_VERSION,
                "state": "RUNNING",
                "external_watchers": {"tsd_cifar10": "stopped-run"},
            }), encoding="utf-8")
            self.assertEqual(load_downstream(path)["external_watchers"], EXTERNAL_WATCHERS)


if __name__ == "__main__":
    unittest.main()
