# CIFAR-10 external multi-view / multi-layer SSL baseline report

Status: implementation validation in progress. This report is restricted to CIFAR-10 and will be updated from Slurm artifacts only.

## Source lock and provenance

| Method | Repository | Branch | Pinned commit | License evidence | Runtime provenance |
|---|---|---|---|---|---|
| FastSSL-Barlow-Twins / FastSSL-VICReg | <https://github.com/kumarkrishna/fastssl> | `main` | `96dfa37be7d46f4814a410affd6269fefab9ec32` | `setup.py` declares MIT; the pinned tree has no root `LICENSE` file | The author objective and small projector are dependency-free ports into the existing Lightning harness; results are harness adaptations, not executions of the FFCV training script |
| FroSSL | <https://github.com/OFSkean/FroSSL> | `main` | `e841c5769944d4d58feeaf0258aff27a9d3934b3` | `setup.py` and source headers declare MIT; the pinned tree has no root `LICENSE` file | The original linear-kernel multiview FroSSL objective, projector, and LARS optimizer are ported into the existing Lightning harness; EMP-FroSSL and all bundled alternative losses are excluded |
| HAI | No author repository was found; paper: <https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Rethinking_the_Augmentation_Module_in_Contrastive_Learning_Learning_Hierarchical_Augmentation_CVPR_2022_paper.html> | n/a | n/a | paper only | Faithful reimplementation will start only after FastSSL/FroSSL validation passes; all under-specified CIFAR choices will be labeled as implementation assumptions |

The machine-readable source lock is `configs/external_baseline_sources.json`. Audit checkouts are detached at the commits above under `/projects/EEG-foundation-model/yinghao/FMCA-AV/external_sources/` and are not vendored into this repository.

The official CVF page exposes only the ten-page paper and an arXiv link; it has no supplementary-material link. The arXiv source archive also contains only that manuscript. Consequently, details referenced by the paper as being in an appendix—but absent from the released manuscript—cannot be treated as known settings. In particular, exact shallow-head kernel/stride/channel choices, projection dimensions, and a CIFAR-specific protocol will be recorded as reimplementation assumptions rather than attributed to the authors.

## Implemented objectives

FastSSL-Barlow-Twins retains the official small two-layer projector and, for more than two views, computes the invariance diagonal estimator against the mean projected view while applying the stronger redundancy penalty to the autocorrelation of that mean. Its redundancy coefficient is (1/d) with (d=256). It does not use the ordinary paired Barlow Twins branch under a new name.

FastSSL-VICReg retains the official linear-complexity estimator: each view is compared with the mean of all other views, with variance and covariance penalties computed for both sides and averaged over views. The two-view control uses the official single non-redundant pair.

FroSSL uses the official `multiview_frossl_loss_func` semantics: coordinate-wise normalization over parents, alignment to the mean view, trace-normalized linear Gram matrices, and the log Frobenius regularizer. The view-dependent invariance weights are 1.4 for two views and 2.0 for eight views. No EMP, MMCR, W-MSE, Barlow Twins, or VICReg component is present.

The pinned FroSSL class always calls `multiview_frossl_loss_func`, including for its two-view case. In that current function the kernel is fixed to linear and the configured `alpha`/`kernel_type` fields are not read; the adapter records those fields for provenance but does not invent an effect for them.

## Formal run matrix

The first formal matrix is three paired seeds for each of:

| Method | Views | Backbone | Projector | Pretraining epochs | State |
|---|---:|---|---|---:|---|
| FastSSL-Barlow-Twins | 2, 8 | CIFAR ResNet-50 | 256-d small projector | 100 | M=2 three-seed full chains passed; M=8 three-seed pretraining active |
| FastSSL-VICReg | 2, 8 | CIFAR ResNet-50 | 256-d small projector | 100 | GPU smoke passed; formal runs pending capacity |
| FroSSL | 2, 8 | CIFAR ResNet-18 | 2048-2048-1024 | 1000 | objective and official-batch scheduler smokes passed; formal runs pending capacity |
| HAI faithful reimplementation | 8 expanded views (four hierarchical pairs) | to be recorded | four stage heads | to be recorded | not started until the first three methods pass |

Each successful checkpoint receives the existing frozen linear probe and weighted kNN evaluation, clean-test backbone/projector covariance spectra, effective rank and collapse diagnostics. Run-level wall time, GPU model, peak memory, encoded views, throughput, parameters, GPU-hours, and supported-operator FLOPs are retained.

The completed supported-operator profiles for FastSSL-Barlow-Twins measure 2.599 GFLOPs per encoded view for both controls. At two views this is 5.199 GFLOPs per parent and 10.398 GFLOPs for the profiled two-parent forward/objective/backward step; at eight views this is 20.794 GFLOPs per parent and 41.587 GFLOPs per profiled step. These are PyTorch-profiler supported-operator counts, not hardware-peak FLOPs.

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

## Partial completed downstream results

These rows are emitted only after the full per-seed pretrain/probe/kNN/diagnostics chain succeeds. They remain partial until all three paired seeds are complete.

| Method | Views | Seed | Linear probe test | kNN test | Backbone effective rank | Backbone numerical rank | Projector effective rank | Mean absolute projector off-diagonal correlation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FastSSL-Barlow-Twins | 2 | 20260821 | 84.63% | 78.98% | 1.269 / 2048 | 575 / 2048 | 1.005 / 256 | 0.9746 |
| FastSSL-Barlow-Twins | 2 | 20260822 | 84.12% | 79.18% | 1.557 / 2048 | 884 / 2048 | 1.011 / 256 | 0.9427 |
| FastSSL-Barlow-Twins | 2 | 20260823 | 83.18% | 79.10% | 1.055 / 2048 | 182 / 2048 | 1.001 / 256 | 0.9797 |

All three M=2 raw chains are complete, but their aggregate and reproduction deviation remain deferred to the existing final aggregation job rather than being calculated by a new side path. The results use the declared harness adaptations (45,000-image SSL split and the harness probe rather than the official 200-epoch Adam probe), so any final gap cannot be attributed to one cause. Although the covariance entropy effective ranks are very low, the backbones retain nontrivial relative numerical ranks and reach 83.18--84.63% linear-probe accuracy; the diagnostics are reported as anisotropy/collapse evidence rather than converted into a binary post-hoc label.

## Validation and results

- `20260811-034147_external-baselines-cpu-tests`: PASS, 4/4 formula/config tests.
- `20260811-034825_external-baselines-regression`: PASS, 52/52 existing and new discovered tests.
- `20260811-035348_external-controller-regression`: PASS, 6/6 scoped formula/config/controller tests.
- `20260811-042332_external-controller-heartbeat-test` / Slurm `934572`: PASS, 2/2 controller tests after making every scheduled 300-second refresh persist its on-disk heartbeat even when no run changes state.
- `20260811-040046_external-c10-fastssl_barlow_twins-v8-smoke` / Slurm `934540`: PASS on A100, finite train/validation losses.
- `20260811-040047_external-c10-fastssl_vicreg-v8-smoke` / Slurm `934541`: PASS on A100, finite train/validation losses.
- `20260811-040047_external-c10-frossl-v8-smoke` / Slurm `934542`: PASS on A100, finite train/validation losses.
- `20260811-040145_external-c10-frossl-scheduler-smoke` / Slurm `934544`: NUMERICAL WARNING. The deliberately tiny batch-8 LARS/scheduler check exited successfully but produced a NaN at its second validation point. Because the official FroSSL recipe uses batch 256, this is recorded as a failed small-batch stress condition rather than evidence about the formal configuration.
- `20260811-040707_external-c10-frossl-scheduler-b256-smoke` / Slurm `934550`: PASS on A100. With the official batch size 256 and real LARS/warmup-cosine schedule, two optimizer steps produced finite train losses 336.98 and 330.43 and finite validation losses 169.76 and 105.05. Peak memory was 9,620 MB.
- `20260811-040548_external-c10-fastssl_barlow_twins-v2-flops` / Slurm `934547`: PASS, 10.398 supported-operator GFLOPs per two-parent training step.
- `20260811-040549_external-c10-fastssl_barlow_twins-v8-flops` / Slurm `934548`: PASS, 41.587 supported-operator GFLOPs per two-parent training step.
- `20260811-040549_external-c10-fastssl_barlow_twins-v2-seed1-pretrain` / Slurm `934549`: PASS, 100/100 epochs on A100-SXM4-40GB. The run encoded 8,960,000 views in 2,200.53 seconds (0.6113 GPU-hours, 4,071.7 views/s), peaked at 6,730 MB, and finished with finite train loss 13.2991 and validation loss 13.8442. Its downstream evaluations remain active/pending and no accuracy is inferred from the pretraining loss.
- `20260811-041050_external-c10-fastssl_barlow_twins-v2-seed2-pretrain` / Slurm `934561`: PASS, 100/100 epochs on A100-PCIE-40GB in 2,444.62 seconds (0.6791 GPU-hours, 3,665.2 views/s), with 6,730 MB peak memory, final train loss 13.1176, and validation loss 13.9410.
- `20260811-041051_external-c10-fastssl_barlow_twins-v2-seed3-pretrain` / Slurm `934562`: PASS, 100/100 epochs on A100-PCIE-40GB in 2,416.30 seconds (0.6712 GPU-hours, 3,708.1 views/s), with 6,730 MB peak memory, final train loss 12.8863, and validation loss 13.9751.
- `20260811-044555_external-c10-fastssl_barlow_twins-v2-seed1-linear-probe` / Slurm `934595`: PASS, 84.63% top-1 and 99.35% top-5 test accuracy; best validation accuracy 84.52%.
- `20260811-045556_external-c10-fastssl_barlow_twins-v2-seed1-knn` / Slurm `934616`: PASS, 78.98% weighted 20-NN test accuracy with a 50,000-sample bank.
- `20260811-045557_external-c10-fastssl_barlow_twins-v2-seed1-diagnostics` / Slurm `934617`: PASS. The clean-test backbone covariance has effective rank 1.269, relative numerical rank 575/2048, and 0.88% dimensions below standard deviation 0.01; the projector has effective rank 1.005, numerical rank 39/256, and mean absolute off-diagonal correlation 0.9746.
- `20260811-050058_external-c10-fastssl_barlow_twins-v2-seed2-linear-probe` / Slurm `934630`: PASS, 84.12% top-1 and 99.44% top-5 test accuracy; best validation accuracy 84.02%.
- `20260811-050058_external-c10-fastssl_barlow_twins-v2-seed2-knn` / Slurm `934631`: PASS, 79.18% weighted 20-NN test accuracy with a 50,000-sample bank.
- `20260811-050059_external-c10-fastssl_barlow_twins-v2-seed2-diagnostics` / Slurm `934632`: PASS. The clean-test backbone covariance has effective rank 1.557 and relative numerical rank 884/2048; the projector has effective rank 1.011, numerical rank 41/256, and mean absolute off-diagonal correlation 0.9427.
- `20260811-050600_external-c10-fastssl_barlow_twins-v2-seed3-linear-probe` / Slurm `934645`: PASS, 83.18% top-1 and 99.11% top-5 test accuracy; best validation accuracy 84.26%.
- `20260811-050601_external-c10-fastssl_barlow_twins-v2-seed3-knn` / Slurm `934646`: PASS, 79.10% weighted 20-NN test accuracy with a 50,000-sample bank.
- `20260811-050631_external-c10-fastssl_barlow_twins-v2-seed3-diagnostics` / Slurm `934647`: PASS. The clean-test backbone covariance has effective rank 1.055 and relative numerical rank 182/2048; the projector has effective rank 1.001, numerical rank 29/256, and mean absolute off-diagonal correlation 0.9797.
- Formal runs, linear probes, kNN, collapse diagnostics, FLOPs, and final numerical summaries: active/pending.

Post-fix aggregate artifacts will be written only after all required actions succeed under `results/postfix/20260809_scientific_correctness_v1/external_multiview_baselines/`. No running or failed result is mixed into the final table.
