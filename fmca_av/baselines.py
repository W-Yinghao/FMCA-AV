"""Lightning SSL baselines sharing the FMCA-AV data and backbone pipeline."""

import copy
from typing import Any, Dict, Tuple

import lightning as L
import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from .backbones import build_backbone
from .models import MLP
from .objectives import trace_score
from .operators import estimate_moments, whitened_cross_operator


def _off_diagonal(matrix: Tensor) -> Tensor:
    size = matrix.shape[0]
    if matrix.shape != (size, size):
        raise ValueError("off-diagonal input must be square")
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


class BaselineSSL(L.LightningModule):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters({"config": config})
        model = config["model"]
        self.method = str(config["experiment"]["method"])
        supported = {
            "simclr", "barlow_twins", "vicreg", "spectral_contrastive",
            "fastsiam", "byol", "moco_v2", "dino", "dcca", "vamp2",
        }
        if self.method not in supported:
            raise ValueError("unsupported baseline method: " + self.method)
        self.backbone = build_backbone(
            str(model.get("backbone", "resnet18_cifar")),
            width=int(model.get("backbone_width", 64)),
        )
        projection_dim = int(model.get("projection_dim", 128))
        self.projector = MLP(
            self.backbone.output_dim,
            projection_dim,
            model.get("projection_hidden_dims", [2048, 2048]),
            str(model.get("activation", "gelu")),
        )
        self.momentum = float(config["objective"].get("momentum", 0.996))
        self.predictor = None
        self.target_backbone = None
        self.target_projector = None
        if self.method in {"fastsiam", "byol"}:
            self.predictor = MLP(
                projection_dim,
                projection_dim,
                model.get("predictor_hidden_dims", [512]),
                str(model.get("activation", "gelu")),
            )
        if self.method in {"byol", "moco_v2", "dino"}:
            self.target_backbone = copy.deepcopy(self.backbone)
            self.target_projector = copy.deepcopy(self.projector)
            for parameter in self.target_backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.target_projector.parameters():
                parameter.requires_grad = False
        if self.method == "moco_v2":
            queue_size = int(config["objective"].get("queue_size", 4096))
            queue = functional.normalize(torch.randn(projection_dim, queue_size), dim=0)
            self.register_buffer("queue", queue)
            self.register_buffer("queue_pointer", torch.zeros(1, dtype=torch.long))
        if self.method == "dino":
            output_dim = int(config["objective"].get("dino_output_dim", 1024))
            self.student_output = nn.Linear(projection_dim, output_dim, bias=False)
            self.teacher_output = copy.deepcopy(self.student_output)
            for parameter in self.teacher_output.parameters():
                parameter.requires_grad = False
            self.register_buffer("teacher_center", torch.zeros(1, output_dim))

    @property
    def config(self) -> Dict[str, Any]:
        return self.hparams["config"]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.target_backbone is not None:
            self.target_backbone.eval()
        if self.target_projector is not None:
            self.target_projector.eval()
        if self.method == "dino":
            self.teacher_output.eval()
        return self

    def _simclr(self, left: Tensor, right: Tensor) -> Tensor:
        temperature = float(self.config["objective"].get("temperature", 0.2))
        values = functional.normalize(torch.cat((left, right), dim=0), dim=1)
        batch = left.shape[0]
        logits = values @ values.transpose(0, 1) / temperature
        logits.fill_diagonal_(float("-inf"))
        targets = torch.arange(2 * batch, device=values.device)
        targets = (targets + batch) % (2 * batch)
        return functional.cross_entropy(logits, targets)

    def _barlow(self, left: Tensor, right: Tensor) -> Tensor:
        epsilon = 1e-4
        left = (left - left.mean(0)) / (left.std(0, unbiased=False) + epsilon)
        right = (right - right.mean(0)) / (right.std(0, unbiased=False) + epsilon)
        correlation = left.transpose(0, 1) @ right / left.shape[0]
        diagonal = (torch.diagonal(correlation) - 1).square().sum()
        coefficient = float(self.config["objective"].get("off_diagonal_weight", 0.0051))
        return diagonal + coefficient * _off_diagonal(correlation).square().sum()

    def _vicreg(self, left: Tensor, right: Tensor) -> Tensor:
        objective = self.config["objective"]
        invariance = functional.mse_loss(left, right)
        left_centered = left - left.mean(0)
        right_centered = right - right.mean(0)
        left_std = torch.sqrt(left_centered.var(0, unbiased=False) + 1e-4)
        right_std = torch.sqrt(right_centered.var(0, unbiased=False) + 1e-4)
        variance = functional.relu(1 - left_std).mean() + functional.relu(1 - right_std).mean()
        denominator = max(1, left.shape[0] - 1)
        left_covariance = left_centered.transpose(0, 1) @ left_centered / denominator
        right_covariance = right_centered.transpose(0, 1) @ right_centered / denominator
        covariance = (
            _off_diagonal(left_covariance).square().sum()
            + _off_diagonal(right_covariance).square().sum()
        ) / left.shape[1]
        return (
            float(objective.get("invariance_weight", 25.0)) * invariance
            + float(objective.get("variance_weight", 25.0)) * variance
            + float(objective.get("covariance_weight", 1.0)) * covariance
        )

    def _spectral_contrastive(self, left: Tensor, right: Tensor) -> Tensor:
        left = functional.normalize(left, dim=1)
        right = functional.normalize(right, dim=1)
        attraction = -2.0 * (left * right).sum(dim=1).mean()
        cross_gram = left @ right.transpose(0, 1)
        repulsion = cross_gram.square().mean() * left.shape[0]
        return attraction + repulsion

    def _operator_objective(self, left: Tensor, right: Tensor) -> Tensor:
        moments = estimate_moments(left, right.unsqueeze(1), centered=True)
        ridge = float(self.config["objective"].get("ridge", 1e-3))
        if self.method == "vamp2":
            return -trace_score(moments, ridge=ridge)
        canonical = whitened_cross_operator(moments, ridge=ridge)
        # DCCA maximizes the nuclear norm of the whitened cross-covariance.
        return -torch.linalg.svdvals(canonical).sum()

    @staticmethod
    def _negative_cosine(prediction: Tensor, target: Tensor) -> Tensor:
        return -functional.cosine_similarity(prediction, target.detach(), dim=1).mean()

    @torch.no_grad()
    def _momentum_update(self) -> None:
        if self.target_backbone is None or self.target_projector is None:
            return
        for online, target in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            target.data.mul_(self.momentum).add_(online.data, alpha=1.0 - self.momentum)
        for online, target in zip(self.projector.parameters(), self.target_projector.parameters()):
            target.data.mul_(self.momentum).add_(online.data, alpha=1.0 - self.momentum)
        for online, target in zip(self.backbone.buffers(), self.target_backbone.buffers()):
            if torch.is_floating_point(target):
                target.data.mul_(self.momentum).add_(online.data, alpha=1.0 - self.momentum)
            else:
                target.data.copy_(online.data)
        for online, target in zip(self.projector.buffers(), self.target_projector.buffers()):
            if torch.is_floating_point(target):
                target.data.mul_(self.momentum).add_(online.data, alpha=1.0 - self.momentum)
            else:
                target.data.copy_(online.data)
        if self.method == "dino":
            for online, target in zip(self.student_output.parameters(), self.teacher_output.parameters()):
                target.data.mul_(self.momentum).add_(online.data, alpha=1.0 - self.momentum)

    @staticmethod
    def _distributed_concat(value: Tensor) -> Tensor:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return value
        gathered = [torch.empty_like(value) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, value.contiguous())
        return torch.cat(gathered, dim=0)

    @staticmethod
    def _distributed_concat_grad(value: Tensor) -> Tensor:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return value
        from torch.distributed.nn.functional import all_gather
        return torch.cat(tuple(all_gather(value.contiguous())), dim=0)

    @torch.no_grad()
    def _enqueue(self, keys: Tensor) -> None:
        keys = self._distributed_concat(keys)
        count = keys.shape[0]
        size = self.queue.shape[1]
        pointer = int(self.queue_pointer)
        if count >= size:
            self.queue.copy_(keys[-size:].transpose(0, 1))
            self.queue_pointer.zero_()
            return
        end = pointer + count
        if end <= size:
            self.queue[:, pointer:end] = keys.transpose(0, 1)
        else:
            first = size - pointer
            self.queue[:, pointer:] = keys[:first].transpose(0, 1)
            self.queue[:, :end - size] = keys[first:].transpose(0, 1)
        self.queue_pointer[0] = end % size

    def _moco(self, left_images: Tensor, right_images: Tensor) -> Tensor:
        query = functional.normalize(self.projector(self.backbone(left_images)), dim=1)
        with torch.no_grad():
            key = functional.normalize(self.target_projector(self.target_backbone(right_images)), dim=1)
        positive = (query * key).sum(dim=1, keepdim=True)
        # Clone because the queue is updated before backward; autograd saves the
        # matrix value needed for the query gradient and rejects in-place reuse.
        negative = query @ self.queue.detach().clone()
        temperature = float(self.config["objective"].get("temperature", 0.2))
        logits = torch.cat((positive, negative), dim=1) / temperature
        labels = torch.zeros(len(query), dtype=torch.long, device=query.device)
        loss = functional.cross_entropy(logits, labels)
        if self.training:
            self._enqueue(key)
        return loss

    def _dino(self, views: Tensor) -> Tensor:
        batch, count = views.shape[:2]
        flattened = views.flatten(0, 1)
        student_projection = self.projector(self.backbone(flattened))
        student_logits = self.student_output(student_projection).reshape(batch, count, -1)
        student_temperature = float(self.config["objective"].get("student_temperature", 0.1))
        student_log_probability = functional.log_softmax(student_logits / student_temperature, dim=-1)
        teacher_count = min(2, count)
        with torch.no_grad():
            teacher_images = views[:, :teacher_count].flatten(0, 1)
            teacher_projection = self.target_projector(self.target_backbone(teacher_images))
            teacher_logits = self.teacher_output(teacher_projection).reshape(batch, teacher_count, -1)
            teacher_temperature = float(self.config["objective"].get("teacher_temperature", 0.04))
            teacher_probability = functional.softmax(
                (teacher_logits - self.teacher_center) / teacher_temperature,
                dim=-1,
            )
            if self.training:
                batch_center = teacher_logits.mean(dim=(0, 1), keepdim=False).unsqueeze(0)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(batch_center, op=torch.distributed.ReduceOp.SUM)
                    batch_center.div_(torch.distributed.get_world_size())
                center_momentum = float(self.config["objective"].get("center_momentum", 0.9))
                self.teacher_center.mul_(center_momentum).add_(batch_center, alpha=1.0 - center_momentum)
        losses = []
        for teacher_index in range(teacher_count):
            for student_index in range(count):
                if student_index == teacher_index:
                    continue
                losses.append(
                    -(teacher_probability[:, teacher_index] * student_log_probability[:, student_index]).sum(dim=1).mean()
                )
        return torch.stack(losses).mean()

    def _shared_step(self, batch: Tuple[Tensor, Tensor, Tensor], split: str) -> Tensor:
        views = batch[0]
        if views.shape[1] < 2:
            raise ValueError("SSL baselines require at least two views")
        if views.shape[1] % 2:
            raise ValueError("matched-view baselines require an even number of views")
        pairs = [(index, index + 1) for index in range(0, views.shape[1], 2)]
        if self.method == "moco_v2":
            loss = torch.stack([self._moco(views[:, left], views[:, right]) for left, right in pairs]).mean()
            encoded_count = views.shape[0] * views.shape[1]
        elif self.method == "dino":
            loss = self._dino(views)
            encoded_count = views.shape[0] * (views.shape[1] + min(2, views.shape[1]))
        else:
            flattened = views.flatten(0, 1)
            projections = self.projector(self.backbone(flattened)).reshape(views.shape[0], views.shape[1], -1)
            target = None
            if self.method == "byol":
                with torch.no_grad():
                    target = self.target_projector(self.target_backbone(flattened)).reshape(views.shape[0], views.shape[1], -1)
            pair_losses = []
            for left_index, right_index in pairs:
                left, right = projections[:, left_index], projections[:, right_index]
                if self.method in {"simclr", "barlow_twins", "vicreg", "spectral_contrastive", "dcca", "vamp2"}:
                    left = self._distributed_concat_grad(left)
                    right = self._distributed_concat_grad(right)
                if self.method == "simclr":
                    pair_loss = self._simclr(left, right)
                elif self.method == "barlow_twins":
                    pair_loss = self._barlow(left, right)
                elif self.method == "vicreg":
                    pair_loss = self._vicreg(left, right)
                elif self.method == "spectral_contrastive":
                    pair_loss = self._spectral_contrastive(left, right)
                elif self.method in {"dcca", "vamp2"}:
                    pair_loss = self._operator_objective(left, right)
                elif self.method == "fastsiam":
                    pair_loss = 0.5 * (
                        self._negative_cosine(self.predictor(left), right)
                        + self._negative_cosine(self.predictor(right), left)
                    )
                elif self.method == "byol":
                    assert target is not None
                    pair_loss = 0.5 * (
                        self._negative_cosine(self.predictor(left), target[:, right_index])
                        + self._negative_cosine(self.predictor(right), target[:, left_index])
                    )
                else:
                    raise RuntimeError("unhandled baseline method")
                pair_losses.append(pair_loss)
            loss = torch.stack(pair_losses).mean()
            encoded_count = flattened.shape[0]
        world_size = torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1
        self.log(f"{split}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{split}/encoded_views", float(encoded_count * world_size), on_step=False, on_epoch=True, reduce_fx="sum", sync_dist=True)
        if split == "val":
            self.log("val_score", -loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def on_train_batch_end(self, outputs: object, batch: object, batch_idx: int) -> None:
        if self.method in {"byol", "moco_v2", "dino"}:
            self._momentum_update()

    def configure_optimizers(self) -> object:
        config = self.config["optimizer"]
        name = str(config.get("name", "sgd"))
        if name == "sgd":
            optimizer: torch.optim.Optimizer = torch.optim.SGD(
                (parameter for parameter in self.parameters() if parameter.requires_grad),
                lr=float(config["learning_rate"]),
                momentum=float(config.get("momentum", 0.9)),
                weight_decay=float(config.get("weight_decay", 0.0)),
            )
        elif name == "adamw":
            optimizer = torch.optim.AdamW(
                (parameter for parameter in self.parameters() if parameter.requires_grad),
                lr=float(config["learning_rate"]),
                weight_decay=float(config.get("weight_decay", 0.0)),
            )
        else:
            raise ValueError("baseline optimizer must be sgd or adamw")
        if str(config.get("scheduler", "cosine")) == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(config.get("scheduler_t_max", self.config["trainer"]["max_epochs"])),
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
