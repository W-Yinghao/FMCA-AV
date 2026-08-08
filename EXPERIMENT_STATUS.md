# FMCA-AV experiment status

Snapshot: 2026-08-09 01:54 CEST. The experiment program is running and is not yet complete.

## Scientific version boundary

The active result version is `20260809_scientific_correctness_v1`. It contains the corrected relative-ridge rule, full-matrix held-out SVD/TSD evaluation, and paired f/g E9 dependence map.

- A run is post-fix only when its artifact carries this exact version string.
- A process launched before the correction remains pre-fix even if it finished later.
- Pre-fix checkpoints are never resumed into a post-fix formal chain.
- Existing files under `results/e1` through `results/e10` are retained as a pre-fix archive. They are not mixed with the new tables.
- Post-fix aggregate assets are written under `results/postfix/20260809_scientific_correctness_v1/`.

The detailed formula changes, validation, and affected-result audit are in `SERVER_CODE_FIX_REPORT.md`.

## Current implementation

- FMCA-AV and the vision experiments use Lightning with the server's existing environment.
- All experiment and test computation is submitted through the Slurm harness.
- The aggregate harness budget is 6 GPUs, with at most 2 GPUs per task.
- Non-ImageNet experiments remain ahead of ImageNet work.
- Scheduler monitoring and orchestration use a 300-second polling interval and `squeue`; no experiment computation runs on the login node.
- Checkpoints, raw runs, logs, datasets, and mutable scheduler state remain server-local and are not committed.

Published correction commits:

- `8ec7283`: relative ridge, held-out spectrum/TSD, and E9 dependence-map fixes.
- `a6016dd`: versioned post-fix formal chains and checkpoint-resume rejection.
- `37fafc0`: version-isolated E4/E5/E7/E9 renderers.
- `58c06f2`: versioned E1 controls.

## Completed post-fix results

### E0/E1 exact and operator recovery

The complete versioned E0/E1 input set and renderer finished successfully:

- primary Gaussian suite: Slurm `930444`;
- high-resource Gaussian extension, 420 records: Slurm `930422`;
- exact discrete channels: Slurm `930446`;
- nonlinear toy recovery: Slurm `930447`;
- finite-sample recovery: Slurm `930448`;
- combined recovery renderer: Slurm `930459`.

The resulting CSV/SVG/caption assets are under `results/postfix/20260809_scientific_correctness_v1/e1/`. The combined Gaussian table has 26 dimension/sample-size rows, the nonlinear table has 15 rows, and the discrete table has 7 family rows. No pre-fix input is accepted by the renderer.

### E1 estimator baselines

Slurm job `930423` completed successfully. It evaluated 450 conditions covering linear CCA, Hermite operator features, validation-tuned Nyström, random-Fourier KICA, and normalized HSIC. The post-fix aggregation contains 45 method/correlation/sample-size rows:

- `results/postfix/20260809_scientific_correctness_v1/e1/estimator_baselines.csv`
- `results/postfix/20260809_scientific_correctness_v1/e1/estimator_baselines.svg`
- `results/postfix/20260809_scientific_correctness_v1/e1/estimator_baselines_caption.txt`

The renderer itself ran through Slurm as job `930431` and completed with exit code 0.

## Active post-fix work

- CIFAR-10/CIFAR-100 TSD severity and image data-processing controls: six one-GPU V100 jobs occupied the full GPU budget at this snapshot. Each newly completed source run records the corrected version and evaluates the held-out spectrum with full-matrix SVD.
- Formal matched-budget SSL: watcher run `20260809-013422_formal-ssl-postfix-state-machine` uses the independent state file `results/orchestration/formal_ssl_postfix_state.json`. It is waiting for capacity and starts from action zero; it cannot consume a legacy state or checkpoint.

The following formal jobs were stopped because their scientific lineage was pre-fix or attached to the legacy state chain: Slurm jobs `929950`, `929951`, `929975`, and `930375`. Their directories remain intact for audit and are not counted as post-fix evidence.

## Required rerun scope

- FMCA-AV, DCCA, and VAMP2 source models trained with the old relative ridge must be trained from scratch.
- SimCLR, Barlow Twins, VICReg, spectral contrastive, FastSiam, BYOL, MoCo v2, DINO, and supervised source checkpoints do not require retraining solely for the ridge correction, but any old spectral calibration/evaluation must be regenerated.
- All E7 held-out TSD calibration/severity outputs must use the full held-out cross matrix.
- All E9 localization, faithfulness, deletion/insertion, and randomization results must be rerun with the paired f/g dependence map.
- Pre-fix and post-fix rows are forbidden from entering the same aggregate table.

## Data and repository boundary

The server dataset root is `/projects/EEG-foundation-model/yinghao/FMCA-AV`. ImageNet uses the authorized existing server copy; ImageNet-100 is derived from the fixed wnid list. No dataset, checkpoint, raw run directory, environment, file hash, SHA, or MD5 is committed.

The next completed experimental batch will be summarized here and pushed to `main`; execution continues until every plan item is completed or has an external blocker recorded.
