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
