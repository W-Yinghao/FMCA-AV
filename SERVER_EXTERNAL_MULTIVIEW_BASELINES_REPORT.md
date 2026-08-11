# CIFAR-10 external multi-view / multi-layer SSL baseline report

Status: implementation validation in progress. This report is restricted to CIFAR-10 and will be updated from Slurm artifacts only.

## Source lock and provenance

| Method | Repository | Branch | Pinned commit | License evidence | Runtime provenance |
|---|---|---|---|---|---|
| FastSSL-Barlow-Twins / FastSSL-VICReg | <https://github.com/kumarkrishna/fastssl> | `main` | `96dfa37be7d46f4814a410affd6269fefab9ec32` | `setup.py` declares MIT; the pinned tree has no root `LICENSE` file | The author objective and small projector are dependency-free ports into the existing Lightning harness; results are harness adaptations, not executions of the FFCV training script |
| FroSSL | <https://github.com/OFSkean/FroSSL> | `main` | `e841c5769944d4d58feeaf0258aff27a9d3934b3` | `setup.py` and source headers declare MIT; the pinned tree has no root `LICENSE` file | The original linear-kernel multiview FroSSL objective, projector, and LARS optimizer are ported into the existing Lightning harness; EMP-FroSSL and all bundled alternative losses are excluded |
| HAI | No author repository was found; paper: <https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Rethinking_the_Augmentation_Module_in_Contrastive_Learning_Learning_Hierarchical_Augmentation_CVPR_2022_paper.html> | n/a | n/a | paper only | Faithful reimplementation will start only after FastSSL/FroSSL validation passes; all under-specified CIFAR choices will be labeled as implementation assumptions |

The machine-readable source lock is `configs/external_baseline_sources.json`. Audit checkouts are detached at the commits above under `/projects/EEG-foundation-model/yinghao/FMCA-AV/external_sources/` and are not vendored into this repository.

## Implemented objectives

FastSSL-Barlow-Twins retains the official small two-layer projector and, for more than two views, computes the invariance diagonal estimator against the mean projected view while applying the stronger redundancy penalty to the autocorrelation of that mean. Its redundancy coefficient is (1/d) with (d=256). It does not use the ordinary paired Barlow Twins branch under a new name.

FastSSL-VICReg retains the official linear-complexity estimator: each view is compared with the mean of all other views, with variance and covariance penalties computed for both sides and averaged over views. The two-view control uses the official single non-redundant pair.

FroSSL uses the official `multiview_frossl_loss_func` semantics: coordinate-wise normalization over parents, alignment to the mean view, trace-normalized linear Gram matrices, and the log Frobenius regularizer. The view-dependent invariance weights are 1.4 for two views and 2.0 for eight views. No EMP, MMCR, W-MSE, Barlow Twins, or VICReg component is present.

The pinned FroSSL class always calls `multiview_frossl_loss_func`, including for its two-view case. In that current function the kernel is fixed to linear and the configured `alpha`/`kernel_type` fields are not read; the adapter records those fields for provenance but does not invent an effect for them.

## Formal run matrix

The first formal matrix is three paired seeds for each of:

| Method | Views | Backbone | Projector | Pretraining epochs | State |
|---|---:|---|---|---:|---|
| FastSSL-Barlow-Twins | 2, 8 | CIFAR ResNet-50 | 256-d small projector | 100 | pending validation |
| FastSSL-VICReg | 2, 8 | CIFAR ResNet-50 | 256-d small projector | 100 | pending validation |
| FroSSL | 2, 8 | CIFAR ResNet-18 | 2048-2048-1024 | 1000 | pending validation |
| HAI faithful reimplementation | 8 expanded views (four hierarchical pairs) | to be recorded | four stage heads | to be recorded | not started until the first three methods pass |

Each successful checkpoint receives the existing frozen linear probe and weighted kNN evaluation, clean-test backbone/projector covariance spectra, effective rank and collapse diagnostics. Run-level wall time, GPU model, peak memory, encoded views, throughput, parameters, GPU-hours, and supported-operator FLOPs are retained.

## Deliberate harness adaptations and source discrepancies

- The existing CIFAR harness reserves 2,500 calibration and 2,500 validation examples, so SSL pretraining uses 45,000 of the 50,000 training images. Official FastSSL and FroSSL results use the full training set.
- The existing downstream protocol is a frozen-backbone 100-epoch SGD/cosine probe with a fixed 45,000/5,000 train/validation split, followed by the untouched CIFAR-10 test set. FastSSL reports a 200-epoch Adam linear evaluation; FroSSL reports SGD with learning rate 0.3 and step decays at epochs 60 and 80. These differences will be included in the reproduction-bias interpretation.
- FastSSL's FFCV input path is replaced by the existing file reader/PIL augmentation path. Crop, jitter, grayscale, flip, and normalization parameters are matched, but decoder/interpolation and input-pipeline timing are not identical.
- The FastSSL repository's provided `cc_*.yaml` files use projector dimension 128, whereas the paper's formal multi-augmentation CIFAR experiment and tabulated result use dimension 256. The formal runs follow the paper's 256-dimensional setting and (1/d) redundancy coefficient.
- FroSSL's paper specifies random resized crop and weight decay (10^{-6}), while the pinned CIFAR YAML disables random resized crop and contains weight decay (10^{-4}). The formal runs follow the paper protocol. This repository discrepancy is not presented as an experimental choice made by the paper.
- The official FroSSL class trains an online classifier on detached backbone features. Omitting that classifier does not change backbone gradients, but removes its small parameter/optimizer/time overhead; cost comparisons will identify this difference.
- All formal jobs use one GPU, so the official per-device objective statistics are unchanged. The adapter supports global differentiable gathering if later invoked under DDP, but DDP is not part of this matrix.

## Official CIFAR-10 comparison targets

- FastSSL-Barlow-Twins, projector 256, 100 epochs: 86.43 ± 0.72% for two views and 92.71 ± 0.19% for eight views (paper Table 7).
- FastSSL-VICReg: the paper plots the multi-view curves but does not tabulate exact two/eight-view CIFAR-10 values; no number will be inferred from the figure.
- FroSSL: 92.8% for two views (paper Table 5). The paper does not report CIFAR-10 four/eight-view values and states that gains from more CIFAR views were negligible.
- HAI does not report a CIFAR-10 experiment, so there is no official CIFAR-10 scalar reproduction target.

## Validation and results

- `20260811-034147_external-baselines-cpu-tests`: PASS, 4/4 formula/config tests.
- `20260811-034825_external-baselines-regression`: PASS, 52/52 existing and new discovered tests.
- `20260811-035348_external-controller-regression`: PASS, 6/6 scoped formula/config/controller tests.
- GPU smokes, formal runs, linear probes, kNN, collapse diagnostics, FLOPs, failures, and final numerical summaries: pending.

Post-fix aggregate artifacts will be written only after all required actions succeed under `results/postfix/20260809_scientific_correctness_v1/external_multiview_baselines/`. No running or failed result is mixed into the final table.
