# FMCA-AV experiment status

Snapshot: 2026-08-09 04:07 CEST. The experiment program is running and is not yet complete.

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
- Formal SSL is capped at 4 concurrent GPUs so E7/E10 and other independent
  non-ImageNet chains can use the remaining global capacity.
- Non-ImageNet experiments remain ahead of ImageNet work.
- Scheduler monitoring and orchestration use a 300-second polling interval and `squeue`; no experiment computation runs on the login node.
- Checkpoints, raw runs, logs, datasets, and mutable scheduler state remain server-local and are not committed.

Published correction commits:

- `8ec7283`: relative ridge, held-out spectrum/TSD, and E9 dependence-map fixes.
- `a6016dd`: versioned post-fix formal chains and checkpoint-resume rejection.
- `37fafc0`: version-isolated E4/E5/E7/E9 renderers.
- `58c06f2`: versioned E1 controls.
- `ff296f7`: complete post-fix E2/E3 mechanism and numerical assets.
- `295f69d`: source-checkpoint-aware E6 result isolation.
- `4e84193`: versioned matched-compute and low-label states.
- `737af06`: versioned ImageNet/E9 states and source provenance.
- `22e2ff7`: restartable post-fix E7 factor suite.
- `55c831d`: persistent post-fix downstream chain.

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

### E7 held-out TSD calibration snapshot

The four analytic calibration configurations were recomputed with the full held-out canonical cross-matrix SVD:

- full calibration: Slurm `930460`;
- feature-dimension-4 diagnostic: Slurm `930461`;
- 20-replicate reliability run: Slurm `930462`;
- high-resource calibration: Slurm `930463`.

The versioned E7 assets are under `results/postfix/20260809_scientific_correctness_v1/e7/`. At this snapshot the calibration table has 26 condition rows and the data-processing table has 20 rows. The severity table is intentionally partial while GPU sources and utility probes continue; it must not yet be used for a final C5 claim.

Renderer job `930476` exposed a partial-data bug when utility accuracy was still missing. The code now keeps such rows in CSV but plots a joint TSD/utility stage only when both values exist. Slurm retry `930477` and newline-normalizing rerender `930480` succeeded.

### E2 conditional-sampling mechanism

The corrected frozen-feature Gaussian variance experiment completed as Slurm `930508`, and renderer `930524` produced 10 post-fix condition rows under `results/postfix/20260809_scientific_correctness_v1/e2/`.

- With parent count fixed, score variance fell from `0.1885` at one view to `0.01667` at 16 views; gradient variance fell from `0.003675` to `1.782e-5`.
- With total views fixed, score variance fell from `0.09852` to `0.02606`; gradient variance fell from `3.548e-4` to `3.711e-5`.

These are mechanism results on frozen Gaussian features, not yet a final C2 representation-learning conclusion; the CIFAR fixed-anchor experiment must use a newly trained post-fix checkpoint.

### E3 objective and numerical controls

The exact numerical ablation (`930509`) and 20-replicate Gaussian/finite estimator controls (`930510`) completed. Renderer `930525` produced 23 exact numerical rows and 900 estimator-control rows across 45 designs; all 900 estimator records succeeded and none reported nonfinite output. Post-fix assets are under `results/postfix/20260809_scientific_correctness_v1/e3/`. CIFAR/ImageNet training rows remain empty until their post-fix source models exist.

## Active post-fix work

- Formal matched-budget SSL: CIFAR-10 FMCA-AV v2 seed1/seed2 completed their first 200-epoch chunks as Slurm `930539` and `930547`. Slurm `930559`, `930603`, `930785`, and `930786` continue naturally. Watcher `20260809-040757_formal-ssl-postfix-cap4` uses the same versioned state and now maintains at most four formal GPUs, leaving two of the six global GPUs for independent chains. No running training was cancelled.
- Full held-out TSD: CIFAR-10 watcher `20260809-015613_postfix-cifar10-tsd-full-severity-sweep` owns the new complete 210-cell matrix. CIFAR-100 watcher `20260809-020626_postfix-cifar100-tsd-full-severity-after-cifar10` waits for that full matrix. The duplicate legacy tail watcher was stopped at 03:40:19 without killing its children; completed versioned cells remain available, and the complete watcher pair guarantees coverage.
- E7 factors: watcher `20260809-030719_postfix-e7-factor-suite` has an independent versioned 54-cell plan covering six datasets, three default-channel seeds, and six channel interventions. Its first two dSprites default training jobs, Slurm `930765` and `930766`, succeeded with current-version checkpoints. Their probes wait for capacity; no old factor checkpoint is reused.
- E10: watcher `20260809-025048_postfix-e10-benchmark-chain-retry` submitted corrected complexity retry `20260809-040629_postfix-e10-complexity`, which succeeded. The preserved first child `930585` failed when the corrected nonfinite-covariance guard raised `ValueError`; commit `37bb056` records that condition as an explicit failed benchmark row and allows the frozen sweep to continue.
- ImageNet-1K is still deferred: watcher `20260809-030218_postfix-imagenet-formal-state-machine` cannot proceed until the complete post-fix small/medium formal SSL watcher succeeds. Per-task ImageNet training remains capped at two GPUs and uses the ImageNet A100/L40S/H100 profiles rather than V100 where possible.
- Persistent downstream watcher `20260809-031837_postfix-complete-downstream-chain` will launch versioned matched-compute, low-label, transfer, localization, and final Slurm CPU renderers only after their post-fix prerequisites finish.

At the snapshot the project harness uses four V100 GPUs in the four active formal SSL jobs listed above; two global slots are available to E7/E10. The capacity regression passed all four tests in Slurm CPU job `930877`. An earlier validation attempt, `930841`, used the system Python without PyTorch and failed before tests; its logs are preserved, and no dependency was installed. Other Slurm jobs under the same Unix account are external to this repository and are neither counted nor modified by the project harness.

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
