# FMCA-AV experiment status

Snapshot: 2026-08-09 15:41 CEST. The experiment program is running and is not yet complete.

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
- The aggregate harness budget is restored to 6 GPUs, with at most 2 GPUs per
  task. Jobs already submitted under the temporary 8-GPU budget are left
  untouched to finish naturally; no successor is admitted until the active
  harness allocation is back within the 6-GPU limit.
- The exhaustive 3,942-action formal queue is paused after its current four-run
  batch.  New GPU allocation is prioritized 70% to E4/E5, 20% to E2/E3, and
  10% to inexpensive E6 evaluation and failure recovery.
- E7 and E9 expansion is paused; E8 and E10 receive no additional compute.
- ImageNet-100 is conditional on stable paired CIFAR-10/100 evidence, and
  ImageNet-1K remains paused.
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
- The current configuration restores the six-GPU aggregate limit without
  cancelling jobs submitted under `d588e6c`.

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

The resource policy and the screening gate were frozen before reading the new
probe results in `configs/experiments/tpami_priority_20260809.json`.  Screening
uses paired seed indices 1--3 at the 200-epoch checkpoint.  Within each method
family, the best method and methods within 1.0 percentage point of it may
advance, with at most two methods per family; GPU-hours, encoded views, and then
method name are preregistered tie-breakers.  Only selected methods receive seeds
4--5 and continuation to the full budget.

- The broad formal successor, TSD expansion, downstream E9/transfer launcher,
  and ImageNet launcher were stopped at the watcher level.  No running Slurm
  training was cancelled and no existing result was removed.
- Formal Slurm jobs `931270`, `931633`, `931699`, and `931835` are the final
  in-flight batch from the broad queue.  Local watcher
  `20260809-121400_priority-drain-formal-current` will reconcile their
  checkpoints and mark that queue `PAUSED` without selecting a successor.
- Eight post-fix FMCA-AV 200-epoch checkpoints were already available: all five
  M=2 seeds and the first three M=8 seeds.  The paired seeds 1--3 are now the
  first E5 priority gate.
- Priority probe jobs `931909`--`931912` were submitted for the first four
  checkpoint/seed pairs, followed by `931928`--`931929` for paired seed 3.
  All six succeeded.  Slurm renderer `932049` froze the 200-epoch gate:
  M=2 reached mean test accuracy `0.85443` at mean `1.7861` GPU-hours, while
  M=8 reached `0.89460` at mean `6.3133` GPU-hours.  Under the preregistered
  one-percentage-point gate, M=8 advances and M=2 does not consume the full
  800-epoch budget.  The raw rows and decision are in
  `results/postfix/20260809_scientific_correctness_v1/e5/priority_fmca_200epoch_gate.*`.
- M=8 paired seeds 1--3 are running their 200-to-400 epoch continuation as
  Slurm `932037`, `932038`, and `932040`.  A persistent priority controller
  will submit their 600/800-epoch chunks followed by linear probe and k-NN.
- Strong-baseline screening started with SimCLR `932041` and VICReg `932042`;
  the bounded controller fills the remaining SimCLR/VICReg/DINO/BYOL paired
  seed cells as the eight-GPU budget releases slots.
- E2 fixed-anchor/fixed-total-view job `932043` and the reduced E3 CIFAR
  logdet/relative-ridge/AMP job `932044` use the two non-E4/E5 slots.  The
  associated continuation, screening, and gate regressions passed as Slurm
  `932045`, `932047`, and `932048`.
- The bounded E4 wave is staged behind the active eight-GPU batch.  CPU Slurm
  `932052` validated its design rules and `932053` instantiated all four models:
  raw-parent and mean differ from the 11,628,864-parameter final model by 179
  parameters, DeepSets by 1, and concat by 609 (maximum relative difference
  `5.24e-5`).  Every condition executes exactly eight backbone forwards per
  parent; raw-parent uses seven conditional views plus one explicit parent.
  Controller compile job `932055` passed.  Watcher
  `20260809-144329_priority-e4-architecture-permutation-wave` will run three
  paired seeds at the matched 200/800 scheduler point, then linear probes and a
  deterministic reverse-view permutation diagnostic for every checkpoint.

After the probe gate, the queue completes the small E4 architecture set and the
CIFAR-10 E5 finalists end-to-end before starting additional methods.  CIFAR-100
is a three-paired-seed confirmation stage.  E2 fixed-parent/fixed-total-view and
the reduced CIFAR E3 numerical checks reuse selected checkpoints.  E6 is limited
to 1%/10% low-label and CIFAR-C.  VOC, COCO, ImageNet robustness, new E7 cells,
and new E9 cells are not scheduled by this priority wave.

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
