"""Frozen E4 architecture designs with matched view/parameter budgets."""

from __future__ import annotations


TARGET_PARAMETERS = 11_628_864
MAX_PARAMETER_DELTA = 609
DESIGNS = ("raw_parent", "mean", "deepsets", "concat")


def override(design: str, seed_index: int) -> dict[str, object]:
    common: dict[str, object] = {
        "experiment": {"name": f"e4-priority-cifar10-{design}-seed{seed_index}"},
        "data": {"num_views": 8, "include_raw_parent": False},
        "trainer": {"max_epochs": 200, "checkpoint_save_top_k": 1},
        "optimizer": {"scheduler_t_max": 800},
    }
    if design == "raw_parent":
        common["data"] = {"num_views": 7, "include_raw_parent": True}
        common["model"] = {"parent_feature_source": "backbone", "parent_aggregation": "raw",
                           "f_head_hidden_dims": [205]}
    elif design == "mean":
        common["model"] = {"parent_feature_source": "backbone", "parent_aggregation": "mean",
                           "f_head_hidden_dims": [205]}
    elif design == "deepsets":
        common["model"] = {"parent_feature_source": "backbone", "parent_aggregation": "deepsets",
                           "aggregator_hidden_dim": 126, "f_head_hidden_dims": [3]}
    elif design == "concat":
        common["model"] = {"parent_feature_source": "backbone", "parent_aggregation": "concat",
                           "f_head_hidden_dims": [31]}
    else:
        raise ValueError(f"unknown E4 priority design {design!r}")
    return common


def encoded_forwards(design: str) -> int:
    value = override(design, 1)
    data = dict(value["data"])
    return int(data["num_views"]) + int(bool(data.get("include_raw_parent", False)))
