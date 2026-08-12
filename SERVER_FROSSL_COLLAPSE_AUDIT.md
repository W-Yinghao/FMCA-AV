# FroSSL CIFAR-10 collapse mechanism audit

Status: existing-checkpoint audit complete; preregistered training controls are running.

## Frozen protocol boundary

The existing result is named **FroSSL-M8 / FMCA-AV augmentation protocol / flattened-view forward**. It is a matched-protocol stress test, not an official FroSSL CIFAR-10 M=8 reproduction: the paper does not report that cell.

The recovery gate was fixed before running the controls. At epoch 200, a control recovers only if all three conditions hold: clean 20-NN accuracy at least 60%, centered backbone effective rank at least 20, and centered backbone top-eigenvalue share at most 0.8.

## Existing M=8 checkpoints

All three checkpoint/config loads had an exact key match and the embedded seed matched the requested seed. The unique-parent batch size was 256, with 8 views and 2,048 encoded views per batch. Evaluation and kNN used the frozen backbone, not the projector.

| Seed | Saved-eval kNN | Saved-eval centered backbone rank | Saved-eval top share | Batch-stat centered rank | Clean-BN-recal centered rank |
|---:|---:|---:|---:|---:|---:|
| 20260821 | 30.57% | 2.005 | 0.7537 | 2.346 | 2.343 |
| 20260822 | 24.68% | 2.486 | 0.5154 | 2.138 | 2.087 |
| 20260823 | 31.31% | 1.396 | 0.9356 | 2.428 | 2.422 |

Augmented BN recalibration, whether flattened or sequential by view, also left centered backbone effective rank between 1.38 and 2.49 and kNN between 24.2% and 31.4%. Projector behavior changes with BN mode, but the backbone does not recover.

Therefore the current three-seed evidence rules out a pure checkpoint-routing, projector-as-feature, unique-batch-size, mean-direction-only, or BN-running-stat-only explanation. It supports systematic encoder-weight low-rank degeneration under this particular M=8 training protocol. It does **not** establish that FroSSL M=8 necessarily collapses on CIFAR-10 under every implementation or augmentation protocol.

## Running controls

| Item | Run ID | Slurm job ID | GPU | Status at 2026-08-12 05:16 CEST |
|---|---|---:|---|---|
| B: sequential forward + strong RRC, 200 epochs | `20260812-050548_frossl-m8-control-b-train` | 937271 | A100 40GB | RUNNING |
| C: flattened forward + no RRC, 200 epochs | `20260812-050549_frossl-m8-control-c-train` | 937272 | A100 40GB | RUNNING |
| Official-code-style M=2 seed 20260841 | `20260812-050549_frossl-official-m2-seed1-train` | 937273 | A100 40GB | RUNNING |
| Official-code-style M=2 seed 20260842 | `20260812-051151_frossl-official-m2-seed2-train` | 937276 | A100 40GB | RUNNING |
| Official-code-style M=2 seed 20260843 | `20260812-051151_frossl-official-m2-seed3-train` | 937277 | A100 40GB | RUNNING |

The M=2 port fixes the pinned repository's two crops, disabled random resized crop, sequential per-view forward, 50k CIFAR training images, gamma 1.0, weight decay 1e-4, FP32, detached online classifier with LR 0.1 and zero weight decay, and its unusual ResNet-18 CIFAR stem padding of 2. It is clearly labeled a harness port rather than execution inside solo-learn.

D (sequential forward + no RRC) will be submitted only if both B and C fail the preregistered gate. Paper-centering and pair-normalized invariance remain deferred objective-level controls and are not mixed into B/C/D.

## Slurm validation

CPU regression run `20260812-050020_frossl-collapse-cpu-regression-v2` passed 10/10 targeted tests. The three evaluation-only audit jobs were Slurm jobs 937268, 937269, and 937270 and all succeeded. No Python test or model computation was run directly on the login node.
