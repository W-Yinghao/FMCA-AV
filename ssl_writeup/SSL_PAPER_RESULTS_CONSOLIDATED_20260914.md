# FMCA-AV self-supervised results — consolidated for the manuscript

2026-09-14. Organized in the order of `ssl_results_section.tex` and
`ssl_results_experimental_details.tex`. Every number below is on the
server in raw form; sources are named per block. Ranges are minimum–
maximum over seeds and are not confidence intervals. All GPU work is
finished; nothing here is provisional.

**Revised 2026-09-15.** §3: the rows for SimCLR, Barlow Twins, VICReg
and MoCo v2 are replaced by runs at per-method learning rates selected
under `BASELINE_LR_SELECTION_PREREG_FROZEN_20260914`; the stage-4
comparison changes sign. §6–§9: the truncated arms are annotated with
the multiview-term code version (`02fa55e`), which — not the data
split — decides the retained-rank branch
(`BIFURCATION_DIAGNOSTIC_RESULTS_20260915.md`). The ridge β = 0
replication with epoch-20 checkpoints completed on 15 September and is
in §5.

Terminology follows the 14 September reorganization: **FMCA-AV (ridge)**
is the practical method (`product_endpoint`: ridge-normalized pairwise
scores, shared Cholesky composition coordinates, exponentially averaged
endpoint target); **truncated whitening** is the alternative training
normalization (`paper_composition`); exact Gram-normalized composition is
population theory and a fixed-representation analysis.

---

## 1. Experimental setup

**Data.** CIFAR-10 primary, CIFAR-100 supplementary. 50,000 training
images; 2,500 reserved for calibration and 2,500 for validation, leaving
**45,000** for pretraining (ridge runs, the refreshed baselines, and the
updated truncated runs). The archived truncated runs reserved 10,000 +
10,000 and pretrained on 30,000.

**Backbone and chain.** CIFAR-adapted ResNet-18 (two basic blocks per
stage; pooled widths 64/128/256/512). Three-state augmentation chain on
stages 2, 3, 4 with 128-dimensional projection heads (512-unit GELU
hidden layer). Root crop 95–100 % area; transitions 20–100 % and 40–100 %
with transition-specific color/grayscale perturbation; all views 32×32.
Four children per transition (first child continues the chain); eight
additional endpoint paths from the root, averaged in feature space
before the aggregation head.

**Ridge training.** 200 epochs, batch 256, SGD lr 0.1, momentum 0.9,
weight decay 1e-4, 10 warm-up epochs then cosine. Coefficients:
adjacent-stage dependence 0.2, composition 128, covariance 1, final-
stage multiview 1. Ridge 1e-3; endpoint-target EMA momentum 0.99.
Validation loss calls also update the EMA buffer (disclosed in the
draft). Trace-based divergence guard at 400 for K = 128.

**Truncated training.** Same architecture and budget; shared truncated
coordinates at relative eigenvalue cutoff 1e-3; the direct endpoint
matrix enters the gradient without an EMA; twelve Newton–Schulz
iterations in the retained eigenspace; auxiliary multiview Grams
detached.

**Evaluation.**
- *Stagewise*: pooled frozen features after each stage, deterministic
  transform, per-coordinate standardization (floor 1e-6), zero-
  initialized multinomial logistic regression, L-BFGS ≤ 200 iterations,
  weight penalty 1e-4, fitted on all 50,000 labeled images, scored on
  the 10,000 test images. Byte-identical code on every arm and every
  baseline (verified by recomputation: same checkpoint, four stages
  bit-identical).
- *Final-stage SGD probe*: unstandardized features, 45,000/5,000
  labeled split, crop+flip, 100 epochs, lr 0.1, cosine, no weight decay.
- *Weighted kNN*: k = 20, cosine, temperature 0.07.
- On the same checkpoints the stagewise probe reads ~0.5 points above
  the SGD probe at the final stage; the two are reported separately.

---

## 2. Representation quality across depth — FMCA-AV (ridge), CIFAR-10

Source: `results/gate1/gate1_20260820_v8/units/product_endpoint__seed{1..5}`;
five seeds for every measure (stage profiles for seeds 4–5 were added on
14 September).

    measure                mean     min–max
    SGD probe, final       84.98    83.63–85.56
    weighted kNN           82.72    81.40–83.52
    stage 1                57.96    57.23–59.50
    stage 2                79.35    77.72–80.69
    stage 3                85.14    83.96–85.61
    stage 4                85.05    83.80–85.58

Accuracy rises sharply between stages 2 and 3 and is flat from 3 to 4.

**Replication.** Three fresh seeds per dataset were trained on 14
September under the current code with checkpoints at epochs 20 and 200
(`gate1_20260914_ridge_ms*`). Every epoch-200 range overlaps the archived
one (CIFAR-10 SGD 83.91–85.30, stage 2 77.72–80.90, stage 4 83.65–85.54;
CIFAR-100 SGD 55.44–56.19, stage 2 54.82–55.35, stage 4 55.72–56.26).

**Epoch 20** (new runs, stage-end taps):

    CIFAR-10   st1 52.25–54.34   st2 61.77–68.09   st3 67.05–70.88   st4 67.62–70.83
    CIFAR-100  st1 29.70–30.14   st2 42.89–43.87   st3 45.15–46.46   st4 44.03–44.82

---

## 3. Comparison with external methods — CIFAR-10, 45,000 images, 200 epochs

Source: `results/baseline_layerwise/ref_*` (FastSiam, SimSiam, BYOL at
SGD lr 0.03) and `reflr_*` (SimCLR and Barlow Twins at lr 0.1, VICReg
and MoCo v2 at lr 0.3). Reference recipes: BatchNorm in projector/
predictor, method-specified projector widths and constants. The four
learning rates were selected per method from {0.03, 0.1, 0.3} on seed 1
by the stage 2–4 mean of the stagewise probe scored on a 5,000-image
holdout of the training pool, criterion frozen before any run
(`BASELINE_LR_SELECTION_PREREG_FROZEN_20260914`,
`BASELINE_LR_SELECTION_RESULTS_20260915.md`); the test set was not
consulted; seeds 2–3 were then run at the selected rate. Three seeds
each. Same stagewise evaluator as §2.

**Epoch 200**

    method          stage 1        stage 2        stage 3        stage 4
    FMCA-AV (ridge) 57.23–59.50    77.72–80.69    83.96–85.61    83.80–85.58
    VICReg (0.3)    55.67–56.46    67.00–67.86    82.84–83.54    88.23–88.61
    Barlow (0.1)    55.30–56.63    67.31–68.25    82.43–82.56    87.40–87.72
    SimCLR (0.1)    54.77–55.32    67.45–68.45    81.81–82.24    86.05–86.36
    MoCo v2 (0.3)   53.80–55.40    64.60–64.89    78.30–78.45    83.42–83.84
    FastSiam        55.49–55.58    64.62–65.42    72.62–72.78    75.35–75.58
    SimSiam         55.19–55.44    64.22–64.87    71.73–72.55    74.11–74.75
    BYOL            54.65–55.24    63.15–63.94    69.89–70.23    72.91–73.95

At stage 2 FMCA-AV's lower bound (77.72) exceeds every method's upper
bound (best 68.45, SimCLR) by 9.27 points. At stage 3 it exceeds
VICReg's upper bound by 0.42. At stage 4 FMCA-AV (83.80–85.58) is
**below** VICReg, Barlow Twins and SimCLR with disjoint ranges (2.65,
1.82 and 0.47 at the nearest edges) and overlaps MoCo v2. The lr-0.03
rows (`SSL_REFERENCE_RECIPES_RESULTS_20260914.md`) had FMCA-AV 0.90
above Barlow Twins at stage 4; that ordering does not survive the
per-method learning rates. These compare complete training systems;
they do not isolate the composition penalty (§5).

**Epoch 20**

    method          stage 1        stage 2        stage 3        stage 4
    FMCA-AV (ridge) 52.25–54.34    61.77–68.09    67.05–70.88    67.62–70.83
    VICReg (0.3)    52.78–54.32    63.82–64.45    74.29–74.93    77.45–77.89
    Barlow (0.1)    53.80–54.41    64.06–64.53    73.54–73.97    76.06–76.98
    SimCLR (0.1)    52.88–54.55    62.70–64.76    71.22–71.65    73.96–74.84
    MoCo v2 (0.3)   49.99–50.68    57.25–57.62    62.15–62.84    63.90–64.92
    FastSiam        51.23–51.51    56.38–57.23    59.11–59.78    58.46–59.84
    SimSiam         50.65–51.57    55.59–56.45    57.92–58.36    56.32–57.38
    BYOL            48.94–50.40    53.64–53.99    55.33–56.18    54.41–55.86

**BYOL** is at its own recipe (4096 → 256 projector and predictor with
BatchNorm, EMA momentum 0.996), three seeds, added 14 September. Its
original runs had collapsed because the shared projector lacked the
BatchNorm BYOL requires (§10); at the reference recipe it trains
(validation cosine 0.83, accuracy rising with depth on every seed) and
is the lowest of the seven at every stage. The old-template projector
with BatchNorm alone gave 73.46–74.75 at stage 4; the wider reference
projector did not improve it at this learning rate.

---

## 4. Final-stage-only multiview control (ridge)

Source: `final_mview` in the same roots; five seeds per dataset.

    dataset    measure          complete FMCA-AV        final-stage only
    CIFAR-10   SGD probe        84.98 (83.63–85.56)     88.97 (88.85–89.19)
    CIFAR-10   stage 2 / 4      79.35 / 85.05           68.56 / 88.78
    CIFAR-100  SGD probe        55.78 (55.44–56.19)     61.34 (61.12–61.96)
    CIFAR-100  stage 2 / 4      55.07 / 56.06           43.91 / 61.14

The control is higher at the final readout and 10–11 points lower at
stage 2 on both datasets: a depth trade-off, not uniform dominance. It
changes the objective structure and is not a composition ablation.

---

## 5. Matched ablations of the ridge formulation — CIFAR-10 (and CIFAR-100 for β = 0)

Source: `gate1_20260823_v8_beta0`, `gate1_20260823_c100_beta0`,
`gate1_20260821_v8_alpha0`, `gate1_20260821_v8_m4`,
`gate1_20260821_v8_parallel`, `gate1_20260822_c100_parallel`. Three
seeds each; each config differs from the full model in exactly the
named field. Profiles computed 14 September.

    arm (CIFAR-10)                 stage 1        stage 2        stage 3        stage 4        SGD probe
    full (β = 128), n = 5          57.23–59.50    77.72–80.69    83.96–85.61    83.80–85.58    83.63–85.56
    β = 0  (no composition)        59.27–60.27    79.91–80.10    83.91–83.97    83.41–83.63    83.55–83.74
    α = 0  (no adjacent scores)    58.64–60.07    80.55–81.11    83.68–84.41    83.64–84.17    83.44–84.14
    M_end = 4 (8 → 4 endpoint)     57.60–60.44    75.05–80.67    83.18–85.28    82.29–85.03    82.11–85.10
    parallel children (star)       58.87–59.80    81.77–82.07    84.29–84.83    84.19–84.83    83.99–84.60

    arm (CIFAR-100)                stage 1        stage 2        stage 3        stage 4        SGD probe
    full, n = 5                    33.08–34.20    54.81–55.48    58.57–59.59    55.75–56.39    55.44–56.19
    β = 0                          33.22–33.91    53.38–53.98    58.06–58.70    54.95–55.30    54.79–55.55
    parallel children (star)       33.41–34.11    55.20–56.06    58.58–59.47    57.03–58.19    56.52–57.18

Removing the composition penalty leaves the CIFAR-10 stage-2 range
inside the full model's; the full model exceeds β = 0 at stage 4 by
0.17 at the nearest edges. On CIFAR-100 the full model exceeds β = 0 at
stages 2–4 by 0.4–0.8 at the nearest edges. The intermediate-stage
advantage over external methods is therefore present without the
composition penalty on CIFAR-10. The parallel-children control is not
below the full model at any stage on either dataset.

**β = 0 replicated with epoch-20/200 checkpoints (15 September;
`RIDGE_BETA0_MILESTONE_RESULTS_20260915.md`).** Three fresh seeds per
dataset, config diff against the archived β = 0 exactly
`checkpoint_milestones`, and against the same-day full-model replication
exactly `loss.beta`. P-C1 confirmed: every epoch-200 range overlaps the
archived β = 0 range (CIFAR-10 SGD 83.42–83.58, stage 2 79.55–79.92,
stage 4 83.19–83.53; CIFAR-100 54.79–55.55, 53.75–53.98, 54.90–55.30).
Same-seed pairs, new runs only (full − β = 0, mean ± SD over three seeds):

    CIFAR-10   epoch 20    full  52.25–54.34   61.77–68.09   67.05–70.88   67.62–70.83
                           β = 0 56.11–56.87   67.11–67.81   68.83–70.10   68.15–69.89
               epoch 200   full  57.46–60.00   77.72–80.90   84.20–85.61   83.65–85.54
                           β = 0 59.78–60.61   79.55–79.92   83.66–83.83   83.19–83.53
                           paired, epoch 200:  −1.73±0.97  −0.66±1.75  +1.21±0.63  +1.46±1.16
    CIFAR-100  epoch 20    full  29.70–30.14   42.89–43.87   45.15–46.46   44.03–44.82
                           β = 0 29.84–30.71   40.73–41.70   41.78–44.14   40.77–43.12
               epoch 200   full  33.09–33.86   54.82–55.35   58.55–59.57   55.72–56.26
                           β = 0 33.22–34.22   53.75–53.98   57.66–58.70   54.90–55.30
                           paired, epoch 20:   −0.32±0.66  +2.27±0.95  +3.14±1.87  +2.69±1.61
                           paired, epoch 200:  −0.11±0.89  +1.20±0.38  +0.89±0.91  +1.00±0.21

On CIFAR-100 the full model is above β = 0 at stages 2–4 with disjoint
ranges at epoch 20 and at epoch 200; on CIFAR-10 β = 0 is above the full
model at stages 1–2 at epoch 20 and the two overlap at stages 3–4.
Two of the six β = 0 units (CIFAR-100 seeds 2, 3) are same-GPU-family
near-copies of the archived units; the other four are independent.

---

## 6. Alternative normalization: truncated whitening

Sources: `gate1_20260910_paper_probe`, `gate1_20260910_paper_v1` (30k);
`gate1_20260913_major45k*` (45k); `results/paper_certificate/`.

**Final-stage SGD probe**

    formulation                 pretraining images   multiview term     CIFAR-10          CIFAR-100
    ridge (n = 5)               45,000               ridge (V7)         83.63–85.56       55.44–56.19
    truncated (n = 3)           30,000               pre-02fa55e / current  82.43–82.84   44.66–45.91
    truncated (n = 3)           45,000               current            78.32–78.47       42.48–44.48

"Multiview term": commit `02fa55e` (11 Sep, 01:52) replaced the
final-stage multiview term of the truncated formulation (ridge-1e-3
trace score on the mean of views → truncated whiteners on pair-specific
Grams, Supplement §1.3). The CIFAR-10 30k units started before it; the
CIFAR-100 30k units and every 45k unit started after it.

**Truncated, 45,000 images, stage-end taps**

    CIFAR-10   epoch 20    st1 55.97–56.57   st2 68.75–70.99   st3 70.40–73.49   st4 69.40–71.96
    CIFAR-10   epoch 200   st1 58.22–58.84   st2 77.96–78.14   st3 80.21–80.94   st4 79.55–79.86
    CIFAR-100  epoch 200   st2 50.10–50.94   st3 52.16–52.45   st4 48.62–50.32

At epoch 200 the truncated 45k arm is above every external method at
stage 2 and below VICReg, Barlow Twins, SimCLR and MoCo v2 at stage 4
(rows of §3, per-method learning rates).

**Training-time retained projected rank** (`train/retained_min`, of 128)

    CIFAR-10   30k   34–47 (ep 0) → 41–50 (ep 2) → 106–115 (ep 50) → 128 (ep 100–199)
    CIFAR-10   45k   34–48 (ep 0) → 17–20 (ep 2) → 18–21 (ep 5–199)
    CIFAR-100  both  34–47 (ep 0) → 17–19 (ep 2) → 18–20 (ep 5–199)

The branch is decided by the multiview-term code version, not by the
split. Under the current term, per-step logging shows the endpoint
level of every 30k and every 45k seed falling from 56–102 to 16–21
about 30 steps into epoch 2 (levels 0 and 1 unchanged); under the
pre-`02fa55e` term the same six units hold 37–48 through epoch 5 on
both splits, matching the archived 30k runs. The 30k-vs-45k rows above
therefore compare two implementations, not two data sizes
(`BIFURCATION_DIAGNOSTIC_RESULTS_20260915.md`).

**Frozen-coordinate operator diagnostics** (retained ranks per state;
endpoint Frobenius norm; relative composition error)

    CIFAR-10, 30k, 10,000 calibration roots   [128,128,128] ×3   norm 9.08–9.29   error 0.160–0.169
    CIFAR-10, 45k,  2,500 roots               [92,115,20] [99,107,21] [68,95,17]
                                              norm 3.64–4.14   error 0.115–0.263
    CIFAR-100, 30k, 10,000 roots              [90,106,19] [95,113,20] [92,110,19]
                                              norm 3.65–3.78   error 0.133–0.184
    CIFAR-100, 45k,  2,500 roots              [100,114,16] [104,121,20] [94,90,1]
                                              norm 3.58 / 3.94 / 0.08   error 0.256 / 0.156 / 3.741

The 2,500-root certificates are not comparable at face value to the
10,000-root ones. The CIFAR-100 seed with endpoint rank 1 decodes at
42.94 % (SGD probe) and has a normal stage profile; its certificate is
an unresolved operator-estimation result.

---

## 7. Objective ablations under truncated whitening — CIFAR-10, 30,000 images

Source: frozen extraction `results_audit_20260912`, reproduced in
`results_reorganized_20260914.tex`; mean ± sample SD, completed runs only.

**Code-version note (2026-09-15).** This table mixes two multiview
terms: "full truncated" and "β = 0" trained before `02fa55e`; "α = 0"
and "two-state chain" after it; "without final-stage multiview" has no
such term under either. The full-vs-β = 0 row pair and the full-vs-no-
multiview pair are within one implementation; full-vs-α = 0 and
full-vs-two-state are confounded by the code change, as are the
four-state, K = 256 and parallel-children rows of the configuration
checks against the nested K = 128 row.

    setting                          stage 1       stage 2       stage 3       stage 4
    full truncated                   59.02±0.99    79.81±0.42    82.78±0.13    82.61±0.22
    without composition (β = 0)      58.52±0.66    78.88±0.40    82.48±0.50    82.33±0.22
    without adjacent-stage (α = 0)   57.83±0.79    77.02±0.34    80.32±0.45    79.82±0.28
    without final-stage multiview    58.30±0.32    76.14±0.06    78.95±0.16    79.00±0.43
    two-state chain                  54.76±1.13    69.51±1.08    71.18±1.16    69.85±1.18

Stage-2 paired increase from composition: 0.93 ± 0.80 points.

**Operators** (relative error; endpoint norm; retained endpoint dims)

    full 0.163±0.005 / 9.19±0.11 / 128      β=0 0.176±0.005 / 9.21±0.14 / 128
    α=0 0.090±0.004 / 4.02±0.03 / 20–21     no multiview 0.691±0.401 / 3.43±4.71 / 2–78
    two-state 0.007±0.001 / 3.94±0.01 / 17–18

**Backbone features** (SGD probe; kNN; effective rank; leading variance %; trace)

    full          82.70±0.23  80.92±0.15  78.92±0.71   9.11±0.14    70.06±1.34
    β=0           82.31±0.34  80.51±0.26  74.20±0.60  10.28±0.07    72.36±1.92
    α=0           79.00±0.15  74.18±0.25   4.97±5.01  75.45±23.47  370.47±325.68
    no multiview  78.07±0.08  75.44±0.15  43.30±2.60  15.93±0.71    14.31±0.18
    two-state     68.93±1.21  57.49±1.68  23.03±2.66  12.78±1.80    86.32±10.49

**Configuration checks** (SGD probe; backbone effective rank)

    nested 3-state, K = 128     82.43–82.84   78.3–79.7
    parallel children           78.39–79.62   21.8–22.8      (error 0.148, norm 3.88, endpoint rank 18)
    four-state chain            75.71–77.05   29.6–31.7
    K = 256                     76.91–78.80    1.01–4.09     (leading variance 74.5–99.9 %, trace 291–70,648)
    endpoint objective only     73.01–74.30    1.03–1.09     (leading variance 99.1–99.7 %)

**Stopping criterion.** Without the final-stage multiview objective the
trace guard (400 at K = 128) tripped at 413.9 (CIFAR-10) and 417.7,
400.2 (CIFAR-100); accuracy summaries include completed runs only. The
guard bounds whitened objective terms and did not react to the K = 256
covariance-scale growth.

---

## 8. Supplementary dataset: CIFAR-100

    formulation                 images    SGD probe (mean, range)        stages 1–4 (means)
    ridge, n = 5                45,000    55.78 (55.44–56.19)            33.50 / 55.07 / 58.91 / 56.06
    ridge β = 0, n = 3          45,000    55.24 (54.79–55.55)            33.55 / 53.70 / 58.30 / 55.07
    final-stage only, n = 5     45,000    61.34 (61.12–61.96)            30.84 / 43.91 / 60.35 / 61.14
    truncated, n = 3            30,000    45.29 (44.66–45.91)            current multiview term
    truncated β = 0, n = 3      30,000    (45.40–50.14)                  stage 2 47.5–50.9 vs full 49.6–49.9; current term
    truncated, n = 3            45,000    43.30 (42.48–44.48)            current multiview term

kNN for ridge: 50.04 (49.81–50.38). The highest ridge stage mean is stage
3, not the final stage.

---

## 9. What each claim may rest on

    claim                                                    evidence                        status
    ridge FMCA-AV learns useful intermediate/final features  §2, five seeds, two datasets    supported
    stagewise ranges exceed the seven external baselines     §3, same evaluator, same budget  supported at stage 2 (9.3 pts) and stage 3 (0.4 pt);
                                                                                              NOT at stage 4: VICReg, Barlow Twins, SimCLR above (disjoint)
    best at the final readout                                §4 control is higher             not claimed
    composition penalty causes the stage-2 advantage         §5 β = 0 inside full's range     not supported on CIFAR-10
    truncated 45k is lower because of the data size          §6                               refuted: the 30k and 45k arms ran different multiview
                                                                                              terms; same term → same branch on both splits
    composition changes operator agreement (truncated 30k)   §7                               supported for that implementation (pre-02fa55e term)
    ridge intrinsically superior to Gram normalization       recipes differ in several ways   not claimed
    archived ridge numbers reproduce                         §2 replication, six ranges       supported

---

## 10. Baseline implementation notes for the appendix

- The original six `ssl_matched` configs shared one objective/model
  block (temperature 0.2, Barlow/VICReg constants, 512-512 projector,
  no normalization); MoCo's queue and momentum were code fallbacks
  (4096, 0.996). The reference-recipe runs in §3 fix projector
  normalization and width and each method's own constants (MoCo
  momentum 0.999, temperature 0.2, queue 4096; SimCLR temperature 0.5;
  Barlow λ 0.0051; VICReg 25/25/1; SimSiam/FastSiam predictor 512).
- The optimizer is SGD, batch 256, cosine with 10 warm-up epochs. The
  learning rate is 0.03 for FastSiam, SimSiam and BYOL, and was selected
  per method from {0.03, 0.1, 0.3} for SimCLR (0.1), Barlow Twins (0.1),
  VICReg (0.3) and MoCo v2 (0.3) on a 5,000-image holdout with the
  criterion frozen first; the test set was not consulted. At lr 0.03
  (kept on record) the same four methods were 4.2–6.0 points lower at
  stage 4. FMCA-AV's own learning rate (0.1) was not re-selected.
- FastSiam was previously unimplemented (the pair loop computed
  independent SimSiam pairs); it now uses the mean-of-other-views target.
  SimSiam and FastSiam at two views coincide.
- BYOL's original collapse (validation cosine 0.98–0.99, accuracy falling
  with depth) is attributable to the missing projector BatchNorm; with
  it, cosine settles at 0.83 and accuracy rises with depth on all seeds,
  both at the old-template width (73.46–74.75) and at the reference
  width (72.91–73.95).
