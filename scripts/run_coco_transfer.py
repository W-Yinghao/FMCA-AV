#!/usr/bin/env python3
"""Lightning COCO detection/instance-segmentation transfer from an FMCA backbone."""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import json
import os
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import ConcatDataset
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms.functional import pil_to_tensor

from fmca_av.config import load_config
from fmca_av.vision_module import VisionFMCAAV
from fmca_av.baselines import BaselineSSL
import xml.etree.ElementTree as ET


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


class COCODataset(Dataset):
    def __init__(self, root: str, split: str, masks: bool) -> None:
        self.root = Path(root); self.split = split; self.masks = masks
        payload = json.loads((self.root / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))
        self.images = sorted(payload["images"], key=lambda item: int(item["id"]))
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload["annotations"]:
            if not annotation.get("iscrowd", 0): grouped[int(annotation["image_id"])].append(annotation)
        self.grouped = grouped

    def __len__(self) -> int: return len(self.images)

    def __getitem__(self, index: int):
        info = self.images[index]; path = self.root / self.split / str(info["file_name"])
        with Image.open(path) as source: image = source.convert("RGB")
        annotations = self.grouped.get(int(info["id"]), [])
        boxes = []; labels = []; masks = []
        for annotation in annotations:
            x, y, width, height = map(float, annotation["bbox"])
            if width <= 1 or height <= 1: continue
            boxes.append((x, y, x + width, y + height)); labels.append(int(annotation["category_id"]))
            if self.masks:
                canvas = Image.new("L", (int(info["width"]), int(info["height"])), 0); draw = ImageDraw.Draw(canvas)
                for polygon in annotation.get("segmentation", []) if isinstance(annotation.get("segmentation"), list) else []:
                    if len(polygon) >= 6: draw.polygon([(polygon[offset], polygon[offset + 1]) for offset in range(0, len(polygon), 2)], fill=1)
                masks.append(torch.from_numpy(__import__("numpy").asarray(canvas, dtype="uint8").copy()))
        target: dict[str, Tensor] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([int(info["id"])], dtype=torch.int64),
        }
        if self.masks:
            target["masks"] = torch.stack(masks) if masks else torch.zeros((0, int(info["height"]), int(info["width"])), dtype=torch.uint8)
        return pil_to_tensor(image).float() / 255.0, target


class VOCDataset(Dataset):
    def __init__(self, root: str, year: str, split: str) -> None:
        self.root = Path(root) / f"VOC{year}"
        self.identifiers = [line.strip() for line in (self.root / "ImageSets" / "Main" / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.class_to_index = {name: index + 1 for index, name in enumerate(VOC_CLASSES)}

    def __len__(self) -> int: return len(self.identifiers)

    def __getitem__(self, index: int):
        identifier = self.identifiers[index]
        with Image.open(self.root / "JPEGImages" / f"{identifier}.jpg") as source: image = source.convert("RGB")
        tree = ET.parse(self.root / "Annotations" / f"{identifier}.xml")
        boxes = []; labels = []
        for item in tree.getroot().findall("object"):
            name = str(item.findtext("name", default="")); bounds = item.find("bndbox")
            if name not in self.class_to_index or bounds is None: continue
            x0 = float(bounds.findtext("xmin", default="0")) - 1.0; y0 = float(bounds.findtext("ymin", default="0")) - 1.0
            x1 = float(bounds.findtext("xmax", default="0")); y1 = float(bounds.findtext("ymax", default="0"))
            if x1 <= x0 or y1 <= y0: continue
            boxes.append((x0, y0, x1, y1)); labels.append(self.class_to_index[name])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return pil_to_tensor(image).float() / 255.0, target


def collate(batch): return tuple(zip(*batch))


class DetectionTransfer(L.LightningModule):
    def __init__(self, detector: nn.Module, learning_rate: float) -> None:
        super().__init__(); self.detector = detector; self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        images, targets = batch; losses = self.detector(list(images), list(targets)); loss = sum(losses.values())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for name, value in losses.items(): self.log(f"train/{name}", value, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD((parameter for parameter in self.parameters() if parameter.requires_grad), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)


def detector_from_checkpoint(config: dict[str, Any], checkpoint: str, task: str, num_classes: int = 91) -> nn.Module:
    source = (BaselineSSL.load_from_checkpoint(checkpoint, config=config, map_location="cpu")
              if config["experiment"].get("method") else
              VisionFMCAAV.load_from_checkpoint(checkpoint, config=config, map_location="cpu"))
    network = source.backbone.network
    backbone = nn.Sequential(OrderedDict([
        ("conv1", network.conv1), ("bn1", network.bn1), ("relu", network.relu), ("maxpool", network.maxpool),
        ("layer1", network.layer1), ("layer2", network.layer2), ("layer3", network.layer3), ("layer4", network.layer4),
    ]))
    backbone.out_channels = 2048  # type: ignore[attr-defined]
    # This backbone exposes only the final ResNet feature map, whereas torchvision's
    # default detector anchors target five FPN levels.  Keep the transfer experiment
    # single-scale and configure every proposal/pooling component for feature map 0.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )
    box_roi_pool = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
    common = {
        "backbone": backbone,
        "num_classes": num_classes,
        "min_size": 320,
        "max_size": 512,
        "rpn_anchor_generator": anchor_generator,
        "box_roi_pool": box_roi_pool,
    }
    if task == "detection":
        return FasterRCNN(**common)
    mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0"], output_size=14, sampling_ratio=2)
    return MaskRCNN(**common, mask_roi_pool=mask_roi_pool)


def box_iou(left: Tensor, right: Tensor) -> Tensor:
    top_left = torch.maximum(left[:, None, :2], right[None, :, :2]); bottom_right = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(2)
    left_area = (left[:, 2:] - left[:, :2]).clamp_min(0).prod(1); right_area = (right[:, 2:] - right[:, :2]).clamp_min(0).prod(1)
    return intersection / (left_area[:, None] + right_area[None, :] - intersection).clamp_min(1e-12)


def mask_iou(left: Tensor, right: Tensor) -> Tensor:
    left_flat = left.reshape(left.shape[0], -1).bool()
    right_flat = right.reshape(right.shape[0], -1).bool()
    intersection = (left_flat[:, None] & right_flat[None, :]).sum(2).float()
    union = (left_flat[:, None] | right_flat[None, :]).sum(2).float()
    return intersection / union.clamp_min(1.0)


def average_precision(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    threshold: float,
    geometry: str,
) -> float:
    classes = sorted({int(label) for target in targets for label in target["labels"]})
    class_values = []
    for category in classes:
        total_gt = sum(int((target["labels"] == category).sum()) for target in targets)
        candidates = []
        for image_index, prediction in enumerate(predictions):
            selected = prediction["labels"] == category
            for item, score in zip(prediction[geometry][selected], prediction["scores"][selected]):
                candidates.append((float(score), image_index, item))
        candidates.sort(key=lambda item: item[0], reverse=True); matched: dict[int, set[int]] = defaultdict(set); tp = []; fp = []
        for _, image_index, item in candidates:
            target = targets[image_index]; selected = torch.nonzero(target["labels"] == category).flatten(); best = -1; best_iou = 0.0
            if len(selected):
                iou_function = box_iou if geometry == "boxes" else mask_iou
                ious = iou_function(item[None], target[geometry][selected])[0]
                position = int(torch.argmax(ious)); best_iou = float(ious[position]); best = int(selected[position])
            success = best_iou >= threshold and best not in matched[image_index]
            tp.append(1.0 if success else 0.0); fp.append(0.0 if success else 1.0)
            if success: matched[image_index].add(best)
        if not candidates or total_gt == 0: continue
        tp_tensor = torch.tensor(tp).cumsum(0); fp_tensor = torch.tensor(fp).cumsum(0)
        recall = tp_tensor / total_gt; precision = tp_tensor / (tp_tensor + fp_tensor).clamp_min(1)
        interpolated = [float(precision[recall >= level].max()) if bool((recall >= level).any()) else 0.0 for level in torch.linspace(0, 1, 101)]
        class_values.append(sum(interpolated) / len(interpolated))
    return sum(class_values) / len(class_values) if class_values else 0.0


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval(); predictions = []; targets = []
    for images, batch_targets in loader:
        output = model([image.to(device) for image in images])
        for prediction, target in zip(output, batch_targets):
            prediction_record = {key: value.cpu() for key, value in prediction.items() if key in {"boxes", "labels", "scores"}}
            target_record = {key: value.cpu() for key, value in target.items() if key in {"boxes", "labels"}}
            if "masks" in prediction and "masks" in target:
                # Downsample only for the custom IoU evaluator; detector outputs and
                # source annotations remain untouched. This bounds smoke-eval memory.
                prediction_record["masks"] = (
                    torch.nn.functional.interpolate(prediction["masks"].float(), size=(64, 64), mode="bilinear", align_corners=False)[:, 0] >= 0.5
                ).cpu()
                target_record["masks"] = (
                    torch.nn.functional.interpolate(target["masks"][:, None].float(), size=(64, 64), mode="nearest")[:, 0] >= 0.5
                ).cpu()
            predictions.append(prediction_record); targets.append(target_record)
    thresholds = [0.5 + 0.05 * index for index in range(10)]
    bbox = {threshold: average_precision(predictions, targets, threshold, "boxes") for threshold in thresholds}
    result = {
        "bbox_AP": sum(bbox.values()) / len(bbox), "bbox_AP50": bbox[0.5],
        "bbox_AP75": bbox[0.75], "evaluated_images": len(targets),
        "evaluation_protocol": "custom_coco_style_101_point_voc_macro",
    }
    if targets and "masks" in targets[0]:
        masks = {threshold: average_precision(predictions, targets, threshold, "masks") for threshold in thresholds}
        result.update({"segm_AP": sum(masks.values()) / len(masks), "segm_AP50": masks[0.5], "segm_AP75": masks[0.75]})
    return result


@torch.inference_mode()
def evaluate_coco(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    annotation_file: Path,
    include_segmentation: bool,
) -> dict[str, object]:
    """Evaluate with the official COCO API without retaining dense masks."""
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    bbox_results: list[dict[str, object]] = []
    segmentation_results: list[dict[str, object]] = []
    image_ids: list[int] = []
    for images, batch_targets in loader:
        outputs = model([image.to(device) for image in images])
        for prediction, target in zip(outputs, batch_targets):
            image_id = int(target["image_id"].reshape(-1)[0])
            image_ids.append(image_id)
            boxes = prediction["boxes"].detach().cpu()
            labels = prediction["labels"].detach().cpu()
            scores = prediction["scores"].detach().cpu()
            for box, label, score in zip(boxes, labels, scores):
                x0, y0, x1, y1 = (float(value) for value in box)
                bbox_results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)],
                    "score": float(score),
                })
            if "masks" in prediction:
                masks = prediction["masks"].detach().cpu()[:, 0] >= 0.5
                for mask, label, score in zip(masks, labels, scores):
                    encoded = mask_utils.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))
                    if isinstance(encoded["counts"], bytes):
                        encoded["counts"] = encoded["counts"].decode("ascii")
                    segmentation_results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "segmentation": encoded,
                        "score": float(score),
                    })

    ground_truth = COCO(str(annotation_file))

    def metrics(kind: str, records: list[dict[str, object]]) -> tuple[float, float, float]:
        if not records:
            return 0.0, 0.0, 0.0
        detections = ground_truth.loadRes(records)
        evaluator = COCOeval(ground_truth, detections, iouType=kind)
        evaluator.params.imgIds = sorted(set(image_ids))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return float(evaluator.stats[0]), float(evaluator.stats[1]), float(evaluator.stats[2])

    bbox_ap, bbox_ap50, bbox_ap75 = metrics("bbox", bbox_results)
    result: dict[str, object] = {
        "bbox_AP": bbox_ap,
        "bbox_AP50": bbox_ap50,
        "bbox_AP75": bbox_ap75,
        "evaluated_images": len(set(image_ids)),
        "evaluation_protocol": "official_pycocotools_cocoeval",
        "annotation_file": str(annotation_file.resolve()),
    }
    if include_segmentation:
        segm_ap, segm_ap50, segm_ap75 = metrics("segm", segmentation_results)
        result.update({"segm_AP": segm_ap, "segm_AP50": segm_ap50, "segm_AP75": segm_ap75})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", required=True); parser.add_argument("--dataset", choices=("coco", "voc"), default="coco")
    parser.add_argument("--task", choices=("detection", "instance_segmentation"), required=True)
    parser.add_argument("--train-images", type=int, default=2000); parser.add_argument("--val-images", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=32); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--resume", default="")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--train-only", action="store_true"); parser.add_argument("--output", default=""); args = parser.parse_args()
    config = load_config(args.config); L.seed_everything(int(config["seed"]) + 13000 + args.seed_offset, workers=True); masks = args.task == "instance_segmentation"
    if args.dataset == "voc" and masks: raise ValueError("VOC transfer currently supports detection, not instance segmentation")
    if args.dataset == "voc":
        train_data = ConcatDataset([VOCDataset(args.root, "2007", "trainval"), VOCDataset(args.root, "2012", "trainval")])
        val_data = VOCDataset(args.root, "2007", "test"); num_classes = 21
    else:
        train_data = COCODataset(args.root, "train2017", masks); val_data = COCODataset(args.root, "val2017", masks); num_classes = 91
    train_data = Subset(train_data, range(min(args.train_images, len(train_data)))); val_data = Subset(val_data, range(min(args.val_images, len(val_data))))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0)
    module = DetectionTransfer(detector_from_checkpoint(config, args.checkpoint, args.task, num_classes), learning_rate=0.0025)
    run_dir = Path(os.environ["FMCA_HARNESS_RUN_DIR"]); checkpoint_dir = run_dir / "artifacts" / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer = L.Trainer(accelerator="gpu", devices=1, max_steps=args.max_steps, max_epochs=-1, precision="32-true", enable_progress_bar=False, log_every_n_steps=5, default_root_dir=str(run_dir))
    trainer.fit(module, train_dataloaders=train_loader, ckpt_path=args.resume or None)
    last_checkpoint = checkpoint_dir / "last.ckpt"; trainer.save_checkpoint(str(last_checkpoint))
    training_result = {"dataset": args.dataset, "task": args.task, "source_checkpoint": str(Path(args.checkpoint).resolve()),
                       "target_steps": args.max_steps, "last_checkpoint": str(last_checkpoint.resolve())}
    temporary = (run_dir / "artifacts" / "detection_train_result.json.tmp")
    temporary.write_text(json.dumps(training_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(run_dir / "artifacts" / "detection_train_result.json")
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": "coco_transfer_train", **training_result}, sort_keys=True) + "\n")
    if args.train_only:
        print(json.dumps(training_result, indent=2)); return 0
    evaluation_device = torch.device("cuda")
    module.detector.to(evaluation_device)
    metrics = (evaluate_coco(
        module.detector,
        val_loader,
        evaluation_device,
        Path(args.root) / "annotations" / "instances_val2017.json",
        include_segmentation=masks,
    ) if args.dataset == "coco" else evaluate(module.detector, val_loader, evaluation_device))
    result = {**training_result, "max_steps": args.max_steps, **metrics}
    default_name = "coco_transfer.json" if args.dataset == "coco" else "voc_detection.json"
    output = Path(args.output) if args.output else Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "artifacts" / default_name
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_suffix(output.suffix + ".tmp"); temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(output)
    with (Path(os.environ["FMCA_HARNESS_RUN_DIR"]) / "metrics.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps({"stage": "coco_transfer", **result}, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
