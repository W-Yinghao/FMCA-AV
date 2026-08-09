# FMCA-AV experiment status

Snapshot: 2026-08-09 10:39 CEST. The experiment program is running and is not yet complete.

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
- The aggregate harness budget is temporarily 8 GPUs, with at most 2 GPUs per task.
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
- `d588e6c`: temporary eight-GPU aggregate budget with a strict two-GPU per-task limit.

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

### E7 factor semantics

The versioned factor suite completed all 54 trainings and all 54 spectral probes across dSprites, Shapes3D, SmallNORB, and MPI3D toy/realistic/real. Every source training artifact carries `20260809_scientific_correctness_v1`; no legacy factor checkpoint is accepted. Slurm summary job `931568` succeeded and produced 16,416 curve rows plus 342 per-factor summary rows under `results/postfix/20260809_scientific_correctness_v1/e7/`.

The complete source package contains three default-channel seeds and six single-seed channel interventions per dataset. It reports top-eigenfunction curves alongside random, bottom-eigenfunction, PCA, unranked-coordinate, and random-rotation controls. These outputs establish the completed factor-probe subtrack; the full E7 claim remains open until the held-out TSD severity matrices finish.

### E2 conditional-sampling mechanism

The corrected frozen-feature Gaussian variance experiment completed as Slurm `930508`, and renderer `930524` produced 10 post-fix condition rows under `results/postfix/20260809_scientific_correctness_v1/e2/`.

- With parent count fixed, score variance fell from `0.1885` at one view to `0.01667` at 16 views; gradient variance fell from `0.003675` to `1.782e-5`.
- With total views fixed, score variance fell from `0.09852` to `0.02606`; gradient variance fell from `3.548e-4` to `3.711e-5`.

These are mechanism results on frozen Gaussian features, not yet a final C2 representation-learning conclusion; the CIFAR fixed-anchor experiment must use a newly trained post-fix checkpoint.

### E3 objective and numerical controls

The exact numerical ablation (`930509`) and 20-replicate Gaussian/finite estimator controls (`930510`) completed. Renderer `930525` produced 23 exact numerical rows and 900 estimator-control rows across 45 designs; all 900 estimator records succeeded and none reported nonfinite output. Post-fix assets are under `results/postfix/20260809_scientific_correctness_v1/e3/`. CIFAR/ImageNet training rows remain empty until their post-fix source models exist.

### E8 Markov direct/composed boundaries

The fresh 10-replicate CPU sweep completed as Slurm `930989` and the version-gated renderer succeeded as `931152`. The source contains 240 exact-chain records and 360 continuous-dynamics records. Post-fix assets under `results/postfix/20260809_scientific_correctness_v1/e8/` contain 16 exact and 12 continuous aggregate rows, plus 960 exact and 1,440 continuous condition rows. The caption records the scientific version and the renderer rejects the retained pre-fix `results/e8` source.

### E10 complexity and single-node DDP scaling

The corrected mixed-precision complexity run `20260809-050132_postfix-e10-complexity` completed all 20 conditions with autocast, GradScaler, and FP32 moment accumulation. The version-gated renderer succeeded as Slurm `931195`, combining it with operator run `930955`, FLOPs run `931093`, the one-GPU DDP point `931076`, and the two-GPU DDP point `931174`.

Both DDP points completed exactly 100 optimizer steps at global parent batch size 128. On the allocated V100 hardware, one GPU encoded 4,325.44 views/s and two GPUs encoded 2,751.88 views/s, giving measured two-GPU scaling efficiency 0.318 relative to the one-GPU point. This unfavorable result is retained without selection or smoothing. A manually submitted duplicate two-GPU run `931173` was stopped after `931174` had already produced the valid artifact; its logs remain preserved and it is not used in the table. The post-fix CSV/SVG/caption assets are under `results/postfix/20260809_scientific_correctness_v1/e10/`.

## Active post-fix work

- Formal matched-budget SSL: CIFAR-10 FMCA-AV v2 seeds1--5 and v8 seed1 completed their first 200-epoch chunks. FMCA-AV v8 seeds2--5 are the current four-run formal wave; one recently completed child is awaiting the next 300-second state-machine handoff. The chained watcher uses the same versioned state and leaves global capacity for independent chains. No running training was cancelled.
- Full held-out TSD: CIFAR-10 watcher `20260809-043414_postfix-cifar10-tsd-full-severity-sweep` resumed a 41-cell manifest and owns the complete 210-cell matrix; 161 cells have been submitted at this snapshot. Its batch limit remains two after the temporary aggregate increase, so it cannot monopolize the eight-GPU budget. CIFAR-100 watcher `20260809-043436_postfix-cifar100-tsd-full-severity-after-cifar10-retry` waits for that full matrix. Both local watchers were restarted in place after the temporary 6-to-8 configuration transition caused one status refresh to reject the intermediate configuration; their manifests and completed Slurm children were preserved. The separate seven-stage image data-processing chain has completed all 35 trainings and 35 utility probes with current-version artifacts. The stopped predecessors and duplicate legacy tail retain their logs and completed versioned cells.
- ImageNet-1K is still deferred: watcher `20260809-045027_postfix-imagenet-formal-state-machine` now gates on `formal_ssl_postfix_state.json` reaching version-matched `SUCCEEDED`, rather than on a hand-off watcher process exiting. An early post-fix CIFAR DDP1 result was migrated to E10 and the ImageNet action index reset to zero; no ImageNet training started. Per-task ImageNet training remains capped at two GPUs and uses the ImageNet A100/L40S/H100 profiles rather than V100 where possible.
- Persistent downstream watcher `20260809-050108_postfix-complete-downstream-chain-e8` will launch versioned matched-compute, low-label, transfer, localization, and final Slurm CPU renderers only after their post-fix prerequisites, including E8 and the version-only completion audit, finish. Its predecessors were stopped before they launched any child task.

At the snapshot the project harness uses five of the temporary eight-GPU budget: three V100s for formal SSL and two V100s for batch-limited CIFAR-10 TSD. The fourth formal slot is between a completed child and the next 300-second state-machine handoff. Per-task capacity remains two GPUs and the independent formal cap remains four. Slurm CPU job `931175` validated the temporary aggregate setting with all four tests passing. Earlier capacity regression `930877`, TSD fairness/dependency migration `931048`/`931085`, and E10/ImageNet state migration `931127`/`931133` passed. An earlier validation attempt, `930841`, used the system Python without PyTorch and failed before tests; its logs are preserved, and no dependency was installed. Other Slurm jobs under the same Unix account are external to this repository and are neither counted nor modified by the project harness.

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
