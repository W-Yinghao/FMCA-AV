# BYOL did not fail; our BYOL was built without the part that holds it up

2026-09-13. Triggered by a direct question about whether BYOL's
training was faulty. It was.

## What was observed

All three matched-budget BYOL seeds collapse, and they do it almost
immediately:

    epoch      0      1      2    ...    199
    val cos  0.985  0.994  0.996         0.980     (val_score = -loss)

Cosine similarity between the online prediction and the EMA target
reaches 0.99 within two epochs and stays there for the remaining 198.
Final val_score 0.9795 / 0.9888 / 0.9880 across seeds. The layer1
representation has effective rank 5.1 of 64 with 43.7% of the variance
in one direction, and probe accuracy DECLINES with depth
(43.8 -> 45.8 -> 42.5 -> 37.6).

## The cause

`fmca_av/models.py:16` defines the `MLP` used for every projector and
predictor in the repo:

    Linear(128,512) -> GELU -> Linear(512,512) -> GELU -> Linear(512,128)

Instantiated and checked: it contains **no BatchNorm1d, no LayerNorm,
no GroupNorm**. BYOL's projector and predictor are specified with
BatchNorm — `Linear -> BatchNorm1d -> ReLU -> Linear` — and the
normalization is not incidental to the method. BYOL has no negatives
and no explicit variance or decorrelation term; the predictor, the EMA
target and the batch statistics are the whole of its collapse
avoidance. Remove the batch statistics and the trivial solution --
every embedding equal -- becomes freely reachable, which is exactly the
solution these three runs found.

## Why only BYOL, when all five baselines share this MLP

Each of the other four carries a collapse-avoidance term in the LOSS,
so the missing normalization costs them nothing structural:

    method        mechanism                     final val_score   outcome
    simclr        InfoNCE negatives                   -2.76       trained
    moco_v2       negatives + memory queue            -4.90       trained
    barlow_twins  off-diagonal decorrelation          -5.03       trained
    vicreg        explicit variance hinge            -13.79       trained
    byol          predictor + EMA + BATCH STATS       +0.98       collapsed

BYOL is the only method in the set whose anti-collapse mechanism is the
component that is missing, and it is the only one that collapsed. That
is the argument for attribution; it is not a controlled test.

## Status of the attribution

The mechanism, the selectivity across five methods and the two-epoch
onset all point one way, but the decisive experiment has not been run.
Re-running BYOL with `BatchNorm1d` in the projector and predictor, at
the same budget and three seeds, is what would settle it.

## What has to change in the record, either way

Two documents currently say BYOL is excluded because it collapsed, and
the wording attributes the failure to the method:

- `PAPER_ESTIMATOR_RESULTS_20260911.md`: "BYOL is excluded: it
  collapsed ... the prereg forbade re-tuning a baseline's recipe, so it
  is reported as not successfully trained rather than beaten."
- `MAJOR_EXPERIMENTS_AND_RESULTS_20260912.md` section 3, same claim.

The prereg's no-re-tuning rule was about not tuning a baseline's
hyperparameters in our favour. Adding the normalization BYOL specifies
is not tuning; it is implementing the method. Omitting it and then
reporting the method as "not successfully trained" states our defect as
a property of the baseline.

Until a BN'd BYOL has run, the honest description is: our BYOL
implementation omits a component the method requires, the run collapsed
as a consequence, and no claim about BYOL's competitiveness -- in
either direction -- is supported by it.

## Scope check

The same normalization-free `MLP` is used by the MAJOR arms
(`hierarchy_module.py:119` and `:260`), so our own method and the
baselines were treated identically in this respect. MAJOR's objective
carries explicit trace and covariance terms, so it is not exposed to
this failure in the way BYOL is. No MAJOR result is implicated by this
diagnosis.

---

# Result, n=3 — the diagnosis is confirmed, on two of the three clauses

Against `BYOL_NORMALIZATION_PREREG_FROZEN_20260913`. Three seeds,
identical budget to the collapsed runs, 45000 training images, 200
epochs, `last.ckpt` throughout.

    arm                  layer1          layer2          layer3          layer4
    BYOL + BatchNorm  [54.70,55.92]  [64.15,64.52]  [70.77,72.04]  [73.46,74.76]
    BYOL no BN        [43.83,44.20]  [44.28,45.81]  [41.47,43.43]  [36.55,38.64]

    P-B1  final val cosine < 0.95    0.8302 / 0.8250 / 0.8316    PASS on all 3
    P-B1  layer1 eff_rank > 20       6.28 / 7.91 / 7.28          FAILED
    P-B2  layer4 > layer1            +19.5 / +18.8 / +18.7       PASS on all 3
          (no-BN, same measure)      -7.3 / -6.8 / -5.2          falls

**P-B1 is half confirmed and half refuted, and the refuted half used a
metric this wave showed to be invalid.** Effective rank does not track
collapse: the collapsed arm reads 58.47 at layer4, higher than simclr,
vicreg and the fixed BYOL, while having the worst probe accuracy of the
four. That threshold should not have been written. See
`EFFECTIVE_RANK_IS_NOT_A_COLLAPSE_TEST_20260913.md`.

What the diagnosis rests on is P-B2 and the cosine: with the BatchNorm
that BYOL specifies, the representation improves monotonically with
depth on every seed and the target agreement stops at 0.83 instead of
0.99. Without it, accuracy falls with depth on every seed. The collapse
was ours.

## P-B3, which carried no prediction

    method          layer4          n
    simclr       [81.90,82.28]      3
    vicreg       [81.44,81.66]      3
    barlow_twins [79.84,80.81]      3
    moco_v2      [78.65,79.30]      3
    byol + BN    [73.46,74.76]      3

A faithfully implemented BYOL is still the weakest of the five matched
baselines, by 5.2 points against the next lowest. The prereg committed
in advance that this is a SEPARATE finding from the diagnosis, and it
is: the original exclusion reached a conclusion that survives — BYOL
does not beat us here — for a reason that did not.

## Standing caveat this wave creates

Only BYOL now has the projector its method specifies. simclr, vicreg,
barlow_twins and moco_v2 still run the normalization-free `MLP`, which
is not the standard recipe for any of them either. They did not
collapse, because each carries an anti-collapse term in its loss, but
their numbers may be understated by an unknown amount. No baseline
comparison in this repo is a comparison against reference
implementations, and that limitation now applies asymmetrically.
