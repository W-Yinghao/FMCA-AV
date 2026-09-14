# MAJOR — every experiment and its result

Consolidated 2026-09-12. Record only: what each experiment is, and what
it returned. No interpretation.

## 0. What counts as a MAJOR experiment

MAJOR is the method: FMCA-AV's trace objective with a Gram-normalized
composition constraint along the view chain (section Eq. 25, supplement
Eq. 6-10). It is defined on the **truncated** estimator — shared
symmetric truncated inverse roots at relative cutoff tau = 1e-3.

The estimator is selected in code, not by config comment:

    hierarchy_module.py   mode = truncated if loss.estimator says so,
                          else truncated iff variant starts with "paper_"
    objective.py:128      whiten_chain_batch(mode=...) defaults to "ridge"

A certificate is a MAJOR certificate only if it came from
`run_paper_certificate.py`, which builds truncated transforms and
stamps `"estimator": "truncated"`. `certificate_evaluation` inside
`run_gate1_unit.py`, `chainspec.py` and `plugin_module.py` all compute
under ridge.

Corpus split: **26+ complete MAJOR units against 286 complete ridge
units**, in cleanly separated result roots. The ridge corpus (V1–V7/V8)
is a loss-form ladder under a different estimator and is not evidence
about MAJOR.

## 1. The ablation grid (the paper's own ladder)

Arms, quoted from the paper. All trained from scratch, 200 epochs,
CIFARResNet width 64, K=128, batch 256, nested view tree,
children_per_edge 4, 3 disjoint seeds.

    arm              what it changes
    composition      Eq. (25) complete            = MAJOR
    beta0            no composition term; keeps the dependence
                     objectives and covariance regularization
    lambda0          removes the aggregation head and score
    alpha0           drops the adjacent-edge scores
    T2               one FMCA-AV pair, no intermediate projection
    endpoint_only    alpha = beta = lambda_mv = 0; endpoint objective
                     plus covariance regulariser

Pre-registration: `PAPER_ESTIMATOR_WAVE_PREREG_FROZEN_20260910` covers
the first five. **endpoint_only has none** and is covered only by
`ENDPOINT_ONLY_APPENDUM_FROZEN_20260912`, frozen after its units
completed but before its numbers were viewed.

### 1a. CIFAR-10 — endpoint linear probe

    arm              n   probe range      knn     eff_rank      trace
    composition      3  [82.43, 82.84]   80.92  [78.3, 79.7]  [ 68.9,    71.5]
    beta0            3  [82.08, 82.70]   80.51  [73.8, 74.9]  [ 70.2,    73.9]
    alpha0           3  [78.87, 79.17]   74.18  [ 1.6, 10.7]  [105.6,   734.1]
    lambda0          2  [78.01, 78.13]   75.44  [41.5, 45.1]  [ 14.2,    14.4]
    endpoint_only    3  [73.01, 74.30]   72.13  [ 1.0,  1.1]  [336.8,   968.6]
    T2               3  [67.56, 69.83]   57.49  [20.0, 24.9]  [ 74.8,    95.4]

eff_rank is the entropy effective rank of the encoder feature
covariance out of 128; trace is that covariance's trace.

### 1b. CIFAR-10 — per-tap probe accuracy (block granularity, 9 taps)

    arm            layer2.1       layer3.1       layer4.0
    composition   79.6 - 80.3    82.6 - 82.9    83.0 - 83.5
    beta0         78.5 - 79.2    82.1 - 83.0    82.4 - 82.9
    alpha0        76.6 - 77.2    80.0 - 80.8    80.0 - 80.6
    lambda0 (n=2) 76.1 - 76.2    78.8 - 79.1    79.1 - 79.5
    endpoint_only 71.6 - 73.7    73.0 - 74.7    75.3 - 76.8
    T2            68.4 - 70.5    69.8 - 71.9    69.7 - 72.1

At layer2.1 composition and beta0 do not overlap. At layer3.1 and
layer4.0 they do.

### 1c. CIFAR-100 — endpoint linear probe

    arm              n   probe range      knn     eff_rank      trace
    lambda0          1  [49.15        ]   43.73  [56.8      ]  [  15.1        ]
    endpoint_only    3  [45.02, 47.78]   44.20  [ 1.0, 60.9]  [   4.4,  2453.1]
    beta0            3  [45.40, 50.14]   38.02  [24.5, 33.0]  [  95.4,   114.3]
    composition      3  [44.66, 45.91]   36.14  [ 1.7, 26.9]  [  52.5,   726.3]
    alpha0           3  [44.62, 45.07]   35.44  [ 3.2, 11.4]  [  79.9,   278.8]
    T2               3  [33.27, 33.88]   21.90  [20.1, 26.4]  [  77.9,    90.9]

### 1d. CIFAR-100 — per-tap probe accuracy

    arm            layer2.1       layer3.1       layer4.0
    composition   49.6 - 49.9    50.4 - 51.4    48.7 - 50.3
    beta0         47.5 - 50.9    50.3 - 54.4    50.1 - 53.9
    alpha0        48.3 - 49.8    50.7 - 51.3    49.6 - 50.1
    endpoint_only 50.0 - 50.5    50.1 - 51.5    53.0 - 53.8
    lambda0 (n=1) 50.6           53.7           53.0
    T2            40.7 - 41.8    40.6 - 41.4    39.2 - 39.9

On CIFAR-100 composition and beta0 overlap at layer2.1, and at
layer4.0 composition's range is the lowest of the non-T2 arms.

### 1e. Failures

    CIFAR-10   lambda0 seed2   divergence guard, edge_trace_sum 413.9 vs 400.0
    CIFAR-100  lambda0 seed1   divergence guard, 417.7 vs 400.0
    CIFAR-100  lambda0 seed2   divergence guard, 400.2 vs 400.0

The bound 400.0 is a hard-coded operational sentinel in
`run_gate1_unit.py`, not a preregistered threshold. All three trips
land within 4.5% of it. `paper_lambda0/seed2` on CIFAR-10 has a second,
cancelled run that is bit-identical to the first on none of their 55
overlapping epochs (H100 vs A100, `deterministic="warn"`).

## 2. Frozen-coordinate certificates

From `run_paper_certificate.py` only. 10000 calibration roots, 10000
evaluation roots, tau = 1e-3, all stamped `estimator: truncated`.

    arm                       retained_ranks     endpoint_norm   rel_closure_err
    composition  C10 s1       [128, 128, 128]        9.287          0.1689
    composition  C10 s2       [128, 128, 128]        9.192          0.1612
    composition  C10 s3       [128, 128, 128]        9.076          0.1602
    beta0        C10 s1-3     [128, 128, 128]        9.07 - 9.34    0.1733 - 0.1818
    alpha0       C10          [ 39-44, 66-71, 20-21]  ~4.0          0.087 - 0.092
    T2           C10          [ 24-39, 17-18]         ~4.2          0.006 - 0.008
    composition  C100 s1      [ 90, 106,  19]          —              —
    composition  C100 s2      [ 95, 113,  20]          —              —
    composition  C100 s3      [ 92, 110,  19]          —              —
    star         C10 s1       [125, 128,  18]        3.880          0.1476
    T4           C10 s1       [ 40, 101, 98, 20]       —              —
    K256         C10 s1       [115, 158,  21]          —              —
    lambda0      C10 s1       [ 88,   1,   2]          —              —
    endpoint_only C10 s1/s2/s3 [97,89,5] / [74,92,28] / [80,100,123]

composition and beta0 reach [128,128,128] on CIFAR-10 only. Every other
arm, and composition itself on CIFAR-100, sits below.

## 3. Like-for-like external baselines (CIFAR-10)

Estimator-independent — these are other methods. Same backbone class,
45000 images, 200 epochs, batch 256, same convex probe on encoder
features before the projection heads.

    method          layer1          layer2          layer3          layer4      n
    MAJOR full  [58.06,60.03]  [79.56,80.29]  [82.63,82.88]  [82.36,82.74]  3
    vicreg      [56.02,56.33]  [67.21,68.06]  [78.16,78.54]  [81.44,81.66]  3
    simclr      [56.12,56.96]  [67.14,67.52]  [77.70,77.95]  [81.90,82.28]  3
    barlow_twins[54.76,55.52]  [66.16,67.33]  [76.98,77.30]  [79.84,80.81]  3
    moco_v2     [54.64,55.48]  [65.52,65.70]  [75.14,75.51]  [78.65,79.30]  3
    byol        [43.83,44.20]  [44.28,45.81]  [41.47,43.43]  [36.55,38.64]  3

BYOL collapsed (validation score 0.9965) and is excluded from
competitiveness claims.  The collapse is an implementation defect on
our side, not a result about BYOL: the shared projector/predictor MLP
has no normalization layer, and BatchNorm is BYOL's whole collapse
avoidance.  The row records what was run and supports no claim about
BYOL in either direction.  See `BYOL_COLLAPSE_DIAGNOSIS_20260913.md`.

## 4. Robustness axes

`aa98113`, CIFAR-10, 3 seeds each. **No pre-registration.** Aggregates
were seen before any reading rule was written; recorded descriptively
in `ROBUSTNESS_AXES_RESULTS_20260912.md`.

    arm    what it changes         n   probe range      eff_rank        trace
    T4     4 levels instead of 3   3  [75.71, 77.05]  [29.6, 31.7]  [107.6,   127.9]
    K256   feature_dim 256         3  [76.91, 78.80]  [ 1.0,  4.1]  [291.4, 70648.0]

K256 per-seed: eff_rank 4.09 / 1.23 / 1.01 of 256, trace 291 / 2367 /
70648, against the composition arm's 69.7.  Probe accuracy nonetheless
rises monotonically with depth (47.9 -> 78.6), so the representation is
anisotropic and scale-inflated rather than non-functional --
`EFFECTIVE_RANK_IS_NOT_A_COLLAPSE_TEST_20260913.md`.  The divergence
guard did not fire: it bounds the objective's trace terms, which are computed on
whitened operators and are scale-invariant.

## 5. The estimator x recipe 2x2

`MAJOR_COMPLETION_WAVE_PREREG_FROZEN_20260912`. CIFAR-10, 3 seeds.
Separates MAJOR's 82.70 from the ridge corpus's 84.98.

    cell  recipe             estimator    n    probe range
    A     paper_composition  truncated    3   [82.43, 82.84]   = MAJOR
    B     product_endpoint   ridge        5   [83.63, 85.56]   = V8
    C     product_endpoint   truncated    3   [81.17, 81.90]
    D     paper_composition  ridge        3   [80.07, 81.34]

Neither C nor D is MAJOR: C runs a recipe the paper forbids (matrix
EMA), D an estimator the supplement calls a different operator
estimate. Cell C's n_calibration is matched to A's 10000; cell B used
2500.

Cell C per-seed: 81.90, 81.79, 81.17; eff_rank 69.3, 69.3, 69.4;
trace 105.6, 105.0, 105.3. Per-tap, cell C seed1: layer2.1 79.2,
layer3.1 82.4, layer4.0 82.9.
Cell D: layer2.1 78.8-80.2, layer3.1 81.3-82.3, layer4.0 81.7-82.1.

## 6. Star negative control

Same prereg, P3. MAJOR's recipe and estimator on a parallel (star) view
tree. Single-factor config difference from the nested arm:
`view_tree.mode` only; children_per_edge, edges and batch_size
identical. In parallel mode each level's children are redrawn from the
original image rather than from the realized parent; endpoint views
stay nested in both modes.

    arm              n   probe range      eff_rank       trace
    star             3  [78.39, 79.62]  [21.8, 22.8]  [ 50.5, 109.4]
    nested (= A)     3  [82.43, 82.84]  [78.3, 79.7]  [ 68.9,  71.5]

Per-tap star: layer2.1 77.2-78.2, layer3.1 79.3-80.6, layer4.0 79.4-80.8.

Training `retained_min` (epoch mean of the per-batch min over levels):

    nested   ep0 41 -> ep25 77 -> ep50 115 -> ep100 128 -> ep199 128
    star s1  ep0 36 -> ep5  16 -> ep25  18 -> pinned at exactly 18.0 through ep199
    star s2  ep5 17.1     star s3  ep5 15.0

Certificate, seed1: retained_ranks [125, 128, 18], endpoint_norm 3.880,
relative_closure_error 0.1476, against nested's [128,128,128], 9.076 -
9.287, 0.1602 - 0.1689.

## 6b. The main run at 45000 images (added 2026-09-14)

`MAJOR_45K_800EPOCH_PREREG_FROZEN_20260913` + budget appendum. Six
units, `paper_composition`, 45000 images, 200 epochs, milestones 20/200.
Full record in `MAJOR_45K_RESULTS_20260914.md`.

    dataset    30k/200ep SGD probe   45k/200ep SGD probe   45k retained_min (ep2 -> ep199)
    CIFAR-10   [82.43, 82.84]        [78.32, 78.47]        17-20 -> 18-21   (30k: 41-50 -> 128)
    CIFAR-100  [44.66, 45.91]        [42.48, 44.48]        17-19 -> 18-20   (30k: 17-19 -> 19-20)

P-M1 and P-M2 refuted: lower on both datasets, disjoint ranges. The
CIFAR-10 45k runs pin retained rank at 18-21 from epoch 2, as every
CIFAR-100 run does; the CIFAR-10 30k runs are the only MAJOR runs on
disk that reach 128.

## 7. Preregistration status of every claim family

    experiment                     prereg                              status
    ablation grid, 5 arms          PAPER_ESTIMATOR_WAVE_FROZEN_0910    preregistered
    endpoint_only                  ENDPOINT_ONLY_APPENDUM_FROZEN_0912  frozen post-hoc,
                                                                       before viewing
    2x2 cells C and D              MAJOR_COMPLETION_WAVE_FROZEN_0912   preregistered
    main run, 45k/200ep            MAJOR_45K_800EPOCH_FROZEN_0913      preregistered
                                   + EPOCH_BUDGET_APPENDUM_0913         (amended, pre-result)
    star control                   same, P3                            preregistered
    T4, K256                       none                                descriptive only
    external baselines             matched-budget wave                 carried over

## 8. Completion state, 2026-09-12 22:30

Every experiment in sections 1-6 is complete. No GPU job is queued or
running. The sweep reports 0 outstanding and 0 blocked, and an audit of
every declared root found 0 anomalies: each complete unit has its block
profile, and each complete truncated unit has its certificate.

    arm                              complete
    ablation grid, CIFAR-10           17/18   (lambda0 seed2 failed)
    ablation grid, CIFAR-100          16/18   (lambda0 seeds 1,2 failed)
    T4                                  3/3   (seed1 in the probe root)
    K256                                3/3   (seed1 in the probe root)
    2x2 cell C                          3/3
    2x2 cell D                          3/3
    star control                        3/3

The three incomplete units are the divergence-guard failures in 1e.
Nothing else is outstanding.

## 9. Known measurement caveats recorded elsewhere

- `"collapsed": false` in every unit record is `test_accuracy < 0.15`,
  not a spectral test.
- The divergence guard bounds whitened objective terms only; nothing
  watches raw representation scale.
- Units are not reproducible run-to-run: `deterministic="warn"` plus a
  launcher spanning A100/H100/L40S.
- CIFAR-100 certificates for beta0 read [87,114,50], [48,34,34],
  [94,104,55] — three seeds, wide spread.
