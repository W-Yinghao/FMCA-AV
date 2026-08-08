# FMCA-AV implementation and experiment status

Snapshot: 2026-08-09 00:35 CEST. The formal experiment program is still running; this document summarizes the implementation and the result assets available at this snapshot. It does not claim that the TPAMI execution plan is complete.

## Implementation

- The FMCA-AV implementation is under `fmca_av/` and uses Lightning 2.6.0 with the server's existing PyTorch 2.8.0+cu128 and torchvision 0.23.0 installation.
- The implementation covers the FMCA objective and spectral estimators, conditional multi-view sampling, aggregation variants, SSL baselines, frozen probes, low-label fine-tuning, robustness evaluation, factor probes, dependence localization, Markov experiments, and complexity profiling.
- Dataset adapters cover CIFAR-10/100, STL-10, Tiny ImageNet-200, ImageNet-100/1K, dSprites, 3D Shapes, SmallNORB, MPI3D, CUB, VOC, COCO, and the configured robustness suites.
- Experiment configuration lives in `configs/`; dataset provenance and layout are recorded in `manifests/datasets.json`; orchestration, evaluation, and result-rendering entry points live in `scripts/`.
- All CPU and GPU experiment work is submitted through the file-based Slurm harness. Aggregate harness capacity is 6 GPUs, while each task is limited to 0, 1, or 2 GPUs. ImageNet work is deliberately scheduled after non-ImageNet work.
- Runtime directories, checkpoints, logs, and the mutable harness job registry remain on the server and are intentionally excluded from Git.

## Data availability

The server-side dataset root is `/projects/EEG-foundation-model/yinghao/FMCA-AV`. Dataset archives and extracted data are not redistributed by this repository. `manifests/datasets.json` records official upstream sources, expected splits, and licenses. ImageNet uses an authorized existing server copy and is not mirrored here; ImageNet-100 is derived from the fixed wnid list in `configs/data/imagenet100_wnids.txt`.

No file hash, SHA, or MD5 is computed or recorded by the experiment workflow.

## Current claim-level evidence

The conclusion cards in `results/claims/` are the authoritative compact summaries for the current result snapshot.

| Claim | Current decision | Narrow result |
| --- | --- | --- |
| C1: dependence spectrum/subspace recovery | PASS | Finite-sample recovery is supported in the reported channel and dimension regimes that meet the projector-error thresholds. |
| C2: variance reduction from conditional multi-sampling | FAIL | The preregistered universal variance-reduction claim is not supported; fixed-budget behavior remains a trade-off. |
| C3: matched-budget representation utility | INCONCLUSIVE | Formal paired downstream contrasts are not yet complete. No blanket superiority claim is supported. |
| C4: semantic factor ordering | INCONCLUSIVE | Existing factor probes do not yet provide corrected positive evidence against all required controls. |
| C5: held-out TSD calibration | INCONCLUSIVE | The required final calibration, processing-chain, and utility tables are not yet complete. |
| C6: Markov spectral composition | PASS | Exact composition is supported for the reported reversible/normal regimes; nonnormal and continuous-dynamics experiments delimit the boundary. |
| C7: localization and faithfulness | INCONCLUSIVE | Localization, faithfulness, and randomization controls are not yet jointly complete. |

Important numerical results already captured in the cards include:

- C1 projector error decreased from 0.3171 to 0.04984 for the asymmetric-cycle condition; the other reported conditions also improved and met their stated regime-specific thresholds.
- C6 reversible/normal chains reached a maximum median spectrum MAE of `1.818e-15`, while the nonnormal counterexample had median spectrum MAE `0.01041`.
- C2's fixed-parent score and gradient variance-ratio confidence bounds remained above one in the reported settings, so the strong noise-reduction criterion failed.

These decisions are provisional with respect to the unfinished formal SSL, transfer, robustness, TSD, and localization waves. Negative and failed-control outcomes are retained rather than filtered out.

## Completed and active formal work

At this snapshot, the formal SSL state machine records 16 successful training actions and no scientific retry queue:

- CIFAR-10 FMCA-AV, 2 views: five seeds at 200 epochs completed.
- CIFAR-10 FMCA-AV, 8 views: five seeds at 200 epochs completed.
- CIFAR-10 FMCA-AV matched-head, 2 views: five seeds at 200 epochs completed.
- CIFAR-10 FMCA-AV DeepSets, 2 views: seed 1 at 200 epochs completed; it finished 35,000 optimizer steps in about 1.89 V100 GPU-hours with approximately 3.85 GB peak allocated memory.
- CIFAR-10 FMCA-AV matched-head, 8 views: five 200-epoch seeds were running at the snapshot.
- The full factor-probe orchestration completed 52/52 scheduled runs successfully.
- The non-ImageNet cross-scale TSD sweep had registered 147/210 CIFAR-100 configurations and was continuing automatically.

ImageNet-1K formal pretraining, ImageNet low-label evaluation, and ImageNet-specific TSD/localization work remain deferred until the non-ImageNet queue is exhausted. Historical 4-GPU scaling attempts are audit-only because the current per-task policy permits at most 2 GPUs; successful 1- and 2-GPU scaling records are retained.

## Result assets

- `results/e1/`: exact, Gaussian, nonlinear, finite-sample, and estimator-baseline recovery tables and figures.
- `results/e2/`: gradient-variance table and figure.
- `results/e4/` and `results/e5/`: aggregation ablations and matched-SSL summaries.
- `results/e6/`: generalization, transfer, and robustness tables and figures.
- `results/e7/`: factor-probe curves and summaries; TSD final assets remain pending.
- `results/e8/`: exact and continuous Markov condition tables and composition figure.
- `results/e10/`: compute-complexity table and scaling figure.
- `results/statistics/`: current confirmatory paired tests with bootstrap intervals, effect sizes, exact sign-flip tests, and Holm adjustment.
- `results/index/`: a 702-run experiment snapshot and an explicit failure atlas. Some failures are expected acceptance tests, stopped superseded jobs, or infrastructure attempts; the atlas preserves their resolution chain.
- `results/orchestration/`: restartable state-machine snapshots and explicit ImageNet deferrals.

## Reproduction boundary

The repository contains code, lightweight configurations, manifests, tests, summarized tables, and vector figures. It intentionally does not contain datasets, Python environments, checkpoints, raw run directories, mutable scheduler state, or the paper PDF. Exact server run artifacts remain under `runs/` and can be summarized again with the included result-building scripts after the formal queue completes.
