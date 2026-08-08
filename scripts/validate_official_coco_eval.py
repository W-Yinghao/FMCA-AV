#!/usr/bin/env python3
"""Validate the official COCOeval dependency and local COCO annotations."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    args = parser.parse_args()
    if not os.environ.get("FMCA_HARNESS_RUN_DIR"):
        raise RuntimeError("COCOeval validation must run through the Slurm harness")
    annotations = Path(args.annotations).resolve()
    ground_truth = COCO(str(annotations))
    candidate = next(
        record for record in ground_truth.anns.values()
        if not record.get("iscrowd", 0) and float(record.get("area", 0.0)) > 0.0
    )
    result = [{
        "image_id": int(candidate["image_id"]),
        "category_id": int(candidate["category_id"]),
        "bbox": [float(value) for value in candidate["bbox"]],
        "score": 1.0,
    }]
    detections = ground_truth.loadRes(result)
    evaluator = COCOeval(ground_truth, detections, iouType="bbox")
    evaluator.params.imgIds = [int(candidate["image_id"])]
    evaluator.params.catIds = [int(candidate["category_id"])]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    ap50 = float(evaluator.stats[1])
    if not math.isfinite(ap50) or ap50 <= 0.0:
        raise RuntimeError(f"official COCOeval validation produced invalid AP50={ap50}")
    image = ground_truth.imgs[int(candidate["image_id"])]
    segmentation = candidate["segmentation"]
    if isinstance(segmentation, list):
        encoded = mask_utils.merge(mask_utils.frPyObjects(segmentation, int(image["height"]), int(image["width"])))
    elif isinstance(segmentation.get("counts"), list):
        encoded = mask_utils.frPyObjects(segmentation, int(image["height"]), int(image["width"]))
    else:
        encoded = dict(segmentation)
    if isinstance(encoded["counts"], bytes):
        encoded["counts"] = encoded["counts"].decode("ascii")
    mask_result = [{
        "image_id": int(candidate["image_id"]),
        "category_id": int(candidate["category_id"]),
        "segmentation": encoded,
        "score": 1.0,
    }]
    mask_detections = ground_truth.loadRes(mask_result)
    mask_evaluator = COCOeval(ground_truth, mask_detections, iouType="segm")
    mask_evaluator.params.imgIds = [int(candidate["image_id"])]
    mask_evaluator.params.catIds = [int(candidate["category_id"])]
    mask_evaluator.evaluate()
    mask_evaluator.accumulate()
    mask_evaluator.summarize()
    segm_ap50 = float(mask_evaluator.stats[1])
    if not math.isfinite(segm_ap50) or segm_ap50 <= 0.0:
        raise RuntimeError(f"official segmentation COCOeval validation produced invalid AP50={segm_ap50}")
    payload = {
        "state": "SUCCEEDED",
        "protocol": "official_pycocotools_cocoeval",
        "annotations": str(annotations),
        "image_id": int(candidate["image_id"]),
        "category_id": int(candidate["category_id"]),
        "bbox_AP50": ap50,
        "segm_AP50": segm_ap50,
    }
    artifact = Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts"
    artifact.mkdir(parents=True, exist_ok=True)
    temporary = artifact / "official_coco_eval_validation.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(artifact / "official_coco_eval_validation.json")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
